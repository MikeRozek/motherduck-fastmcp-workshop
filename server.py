"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger
from mcp.types import InputRequiredResult
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
log = get_logger("sf311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

GroupBy = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
ServiceArg = Annotated[
    str | None,
    Field(description="Exact service name from dataset_info, such as 'Graffiti Public'. Omit for all services."),
]
NeighborhoodArg = Annotated[
    str | None,
    Field(description="Exact official neighborhood name from dataset_info, such as 'Mission'. Omit for all neighborhoods."),
]


HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def mask_address(address: str) -> str:
    """Cut an address down to its block: "994 HARRISON ST, SAN FRANCISCO, ..." -> "900 block of HARRISON ST".

    Addresses without a leading house number, such as intersections, are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address or "")
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """Return a request as a dict with its address masked to the block, safe to show the model."""
    row = request.as_dict()
    row["address"] = mask_address(row["address"])
    return row


def _resolve(value: str | None, choices: list[str], kind: str) -> str | None:
    """Return the real name for `value`, matching case-insensitively, or raise a ToolError with suggestions."""
    if value is None:
        return None
    by_lower = {choice.lower(): choice for choice in choices}
    match = by_lower.get(value.strip().lower())
    if match:
        return match
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} '{value}'.{hint} Call dataset_info for the full list of {kind} names.")


def _matching(service: str | None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    service = _resolve(service, SERVICES, "service")
    neighborhood = _resolve(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: ServiceArg = None,
    neighborhood: NeighborhoodArg = None,
    group_by: Annotated[
        GroupBy | None,
        Field(description="Split the count by this field. Omit to get one total."),
    ] = None,
) -> int | dict[str, int]:
    """Count 311 requests this week, optionally filtered and grouped.

    Use this for any "how many" or "which X had the most" question: for
    example, the neighborhood with the most requests for a service is
    count_requests(service=..., group_by="neighborhood"), and its first key.

    Returns a single number without group_by. With group_by, returns a dict of
    group to count, largest first, at most 50 groups; requests with no value
    for that field are left out.

    Use median_close_hours for how fast requests close, and search_requests
    only when you need to see individual requests.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def count_open(
    service: ServiceArg = None,
    neighborhood: NeighborhoodArg = None,
    group_by: Annotated[
        Literal["neighborhood", "service", "agency"] | None,
        Field(description="Split the open count by this field. Omit to get one total."),
    ] = None,
) -> dict:
    """Count requests opened this week that are still open (not yet closed).

    Use this for "how many are still open", "what is unresolved", or "what
    share is still open" questions, such as count_open(service="Tree Maintenance").

    Without group_by, returns {"open": n, "opened_this_week": total, "open_share": fraction 0-1},
    so you can quote both the backlog and the rate. With group_by, returns a dict
    of group to open count, largest first, at most 50 groups.

    Use count_requests for all requests regardless of status, and
    median_close_hours for how fast closed requests were resolved.
    """
    rows = _matching(service, neighborhood)
    still_open = [r for r in rows if r.status == "Open"]
    if group_by is None:
        return {
            "open": len(still_open),
            "opened_this_week": len(rows),
            "open_share": round(len(still_open) / len(rows), 3) if rows else None,
        }
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in still_open if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


BOUNDARY_KEY = "boundary"


def _parts(local_name: str) -> list[str]:
    """Official neighborhoods a local name overlaps, most requests first."""
    counts = Counter(r.neighborhood for r in REQUESTS if r.local_name == local_name and r.neighborhood)
    return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _count_local(local_name: str, part: str | None) -> int:
    """Requests in a local area, optionally only those inside one official neighborhood."""
    return sum(1 for r in REQUESTS if r.local_name == local_name and (part is None or r.neighborhood == part))


@mcp.tool(annotations={"readOnlyHint": True})
def count_in_neighborhood(
    name: Annotated[
        str,
        Field(description="An official neighborhood, such as 'Mission', or a local name people use, such as 'Lower Nob Hill'."),
    ],
    ctx: Context,
    boundary: Annotated[
        str | None,
        Field(
            description=(
                "Only for a local name that spans several official neighborhoods: one of those "
                "neighborhoods to count just that part, or 'all' for the whole area. Omit to ask the user."
            )
        ),
    ] = None,
) -> int | str | InputRequiredResult:
    """Count this week's 311 requests in a neighborhood, by official or local name.

    Use this when the user names a place in their own words, such as "Lower Nob
    Hill" or "Duboce Triangle". An official name is counted directly. A local
    name inside one official neighborhood is counted directly too. A local name
    that spans several official neighborhoods asks the user which part to count,
    unless you pass boundary.

    Returns the count as a number, or a short message if the user declined to
    choose; in that case nothing was counted, and you can call again with
    boundary="all" or a specific neighborhood. For other filters or groupings
    by official neighborhood, use count_requests.
    """
    key = name.strip().lower()
    official = {n.lower(): n for n in NEIGHBORHOODS}.get(key)
    if official:
        return sum(1 for r in REQUESTS if r.neighborhood == official)

    local = {n.lower(): n for n in LOCAL_NAMES}.get(key)
    if local is None:
        suggestions = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown neighborhood '{name}'.{hint} Call dataset_info for the official names.")

    parts = _parts(local)
    if len(parts) <= 1:
        return _count_local(local, None)

    everything = f"All of {local}"
    if boundary is None:
        choice = read_choice(ctx, BOUNDARY_KEY)
        if choice is None:
            return ask_choice(
                BOUNDARY_KEY,
                f"{local} spans {len(parts)} official neighborhoods. Which part should I count?",
                parts + [everything],
            )
        if not choice.accepted:
            return (
                f"Nothing was counted: the user did not choose which part of {local} to count. "
                f"Call again with boundary='all' or one of: {', '.join(parts)}."
            )
        boundary = choice.value

    picked = boundary.strip()
    if picked.lower() in ("all", everything.lower()):
        return _count_local(local, None)
    part = {p.lower(): p for p in parts}.get(picked.lower())
    if part is None:
        raise ToolError(f"'{boundary}' is not part of {local}. Use 'all' or one of: {', '.join(parts)}.")
    return _count_local(local, part)


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: ServiceArg = None,
    group_by: Annotated[
        Literal["neighborhood", "agency"] | None,
        Field(description="Split the median by neighborhood or agency. Omit to get one overall median."),
    ] = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests this week.

    Use this for "how fast" or "which agency/neighborhood is slowest" questions.
    Only closed requests count; open ones are ignored.

    Returns hours rounded to 0.1, or null if nothing matching has closed. With
    group_by, returns a dict of group to median hours, fastest first, leaving
    out groups with fewer than 10 closed requests (so a missing group means
    too little data, not zero).

    Use count_requests for volumes, not this tool.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN}
    ranked = sorted(medians.items(), key=lambda item: (item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: ServiceArg = None,
    neighborhood: NeighborhoodArg = None,
    status: Annotated[
        Literal["Open", "Closed"] | None,
        Field(description="Only requests with this status. Omit for both."),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=MAX_ROWS, description="How many requests to return, 1 to 50.")] = 20,
) -> list[dict]:
    """List individual 311 requests, newest first, with all their fields.

    Use this to show examples or read details such as addresses, subtypes, or
    timestamps. Do not use it to count or rank: it returns at most 50 rows,
    so totals from it are wrong. Use count_requests or median_close_hours for
    any number.

    Each row has request_id, opened, closed, status, service, subtype, agency,
    neighborhood, local_name, address, and source. Addresses are given only to
    the block, such as "900 block of HARRISON ST"; exact house numbers are
    never available. Pass a request_id to
    get_request to fetch one request again.
    """
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(
    request_id: Annotated[str, Field(description="The request_id of a 311 request, as returned by search_requests.")],
) -> dict:
    """Fetch one 311 request by its request_id, with all its fields.

    The address is given only to the block, such as "900 block of HARRISON ST".

    Use this when you already have an id. To find requests, use
    search_requests; to count them, use count_requests.
    """
    log.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id '{request_id}'. Use search_requests to find real request ids.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
