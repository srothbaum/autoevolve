"""
Visualize the lineage tree of an evo-db database instance.

Usage:
    uv run python visualize_lineage_tree.py <path_to_evo_db.json> [--output <path_to_output.png>]
    uv run python visualize_lineage_tree.py <path_to_evo_db.json> --format svg
    uv run python visualize_lineage_tree.py <path_to_evo_db_1.json> <path_to_evo_db_2.json>  # side-by-side comparison
"""

import argparse
import json
import os
import sys
from html import escape

try:
    import graphviz
except ImportError:
    print("ERROR: graphviz package required. Install with: pip install graphviz", file=sys.stderr)
    print("Also need graphviz system package: brew install graphviz", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

ISLAND_COLORS = [
    "#4A90D9",  # blue
    "#D94A4A",  # red
    "#4AD98A",  # green
    "#D9A54A",  # orange
    "#9B59B6",  # purple
]

ISLAND_LIGHT = [
    "#D6E4F0",  # light blue
    "#F0D6D6",  # light red
    "#D6F0E0",  # light green
    "#F0E4D6",  # light orange
    "#E8D5F5",  # light purple
]

BEST_COLOR = "#FFD700"  # gold
CRASH_COLOR = "#888888"
MIGRATED_BORDER = "#FF8C00"  # dark orange


def load_db(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def is_migrated(exp: dict) -> bool:
    return exp.get("description", "").startswith("[migrated from island")


def _html_label(eid: str, metric_line: str, desc: str) -> str:
    """Build an HTML-like label safe from special characters."""
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">'
        f'<TR><TD><B>{escape(eid[:8])}</B></TD></TR>'
        f'<TR><TD>{escape(metric_line)}</TD></TR>'
        f'<TR><TD><FONT POINT-SIZE="8">{escape(desc)}</FONT></TD></TR>'
        f'</TABLE>>'
    )


def build_graph(db: dict, name: str, fmt: str = "png") -> graphviz.Digraph:
    experiments = db["experiments"]
    best_id = db.get("best_id")
    archive_ids = set(db.get("archive", []))

    # Find the global best val_bpb
    best_bpb = float("inf")
    for exp in experiments.values():
        if exp.get("status") == "success":
            bpb = exp.get("val_bpb", float("inf"))
            best_bpb = min(best_bpb, bpb)

    g = graphviz.Digraph(
        name=name,
        format=fmt,
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "fontsize": "11",
            "bgcolor": "#FAFAFA",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "label": f'<<B>{escape(name)}</B><BR/>best={best_bpb:.5f} | {len(experiments)} experiments>',
            "labelloc": "t",
            "labeljust": "l",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "style": "filled",
            "shape": "plaintext",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
            "color": "#666666",
        },
    )

    # Track which nodes are migrated (exclude from rank grouping)
    migrated_ids = set()

    # Group nodes by generation for rank alignment
    generations = {}
    for eid, exp in experiments.items():
        if is_migrated(exp):
            migrated_ids.add(eid)
        else:
            gen = exp.get("generation", 0)
            generations.setdefault(gen, []).append(eid)

    # Add nodes
    for eid, exp in experiments.items():
        island = exp.get("island", 0)
        status = exp.get("status", "success")
        migrated = eid in migrated_ids
        bpb = exp.get("val_bpb", None)
        params = exp.get("num_params_M", None)

        # Truncate description
        desc = exp.get("description", "")
        if desc.startswith("[migrated from island"):
            desc = desc.split("] ", 1)[-1] if "] " in desc else desc
        if len(desc) > 50:
            desc = desc[:47] + "..."

        # Build label
        if status == "success" and bpb is not None:
            metric_line = f"bpb={bpb:.5f}"
            if params is not None:
                metric_line += f"  {params:.0f}M"
            label = _html_label(eid, metric_line, desc)
        else:
            label = _html_label(eid, "CRASH", desc)

        # Styling
        if status == "crash":
            fillcolor = CRASH_COLOR
            fontcolor = "white"
        elif eid == best_id:
            fillcolor = BEST_COLOR
            fontcolor = "black"
        else:
            fillcolor = ISLAND_LIGHT[island % len(ISLAND_LIGHT)]
            fontcolor = "#333333"

        border_color = MIGRATED_BORDER if migrated else ISLAND_COLORS[island % len(ISLAND_COLORS)]
        penwidth = "2.5" if eid == best_id else ("2.0" if migrated else "1.2")

        # Use Mrecord-style box via HTML table with border
        box_label = (
            f'<<TABLE BORDER="{penwidth}" CELLBORDER="0" CELLSPACING="2" CELLPADDING="4"'
            f' BGCOLOR="{fillcolor}" COLOR="{border_color}" STYLE="ROUNDED">'
            f'<TR><TD><B><FONT COLOR="{fontcolor}">{escape(eid[:8])}</FONT></B></TD></TR>'
        )
        if status == "success" and bpb is not None:
            metric_line = f"bpb={bpb:.5f}"
            if params is not None:
                metric_line += f"  {params:.0f}M"
            box_label += f'<TR><TD><FONT COLOR="{fontcolor}" POINT-SIZE="9">{escape(metric_line)}</FONT></TD></TR>'
        else:
            box_label += f'<TR><TD><FONT COLOR="{fontcolor}" POINT-SIZE="9">CRASH</FONT></TD></TR>'
        box_label += f'<TR><TD><FONT COLOR="{fontcolor}" POINT-SIZE="7">{escape(desc)}</FONT></TD></TR>'
        box_label += '</TABLE>>'

        g.node(
            eid,
            label=box_label,
        )

    # Add edges
    for eid, exp in experiments.items():
        parent_id = exp.get("parent_id")
        if parent_id and parent_id != "none" and parent_id in experiments:
            migrated = eid in migrated_ids
            parent_bpb = experiments[parent_id].get("val_bpb", float("inf"))
            child_bpb = exp.get("val_bpb", float("inf"))

            # Color edge by improvement
            if exp.get("status") == "success" and child_bpb < parent_bpb:
                edge_color = "#2ECC71"  # green = improvement
                edge_label = f"{child_bpb - parent_bpb:+.5f}"
            elif exp.get("status") == "crash":
                edge_color = CRASH_COLOR
                edge_label = ""
            else:
                edge_color = "#E74C3C"  # red = regression
                edge_label = ""

            style = "dashed" if migrated else "solid"

            g.edge(
                parent_id, eid,
                color=edge_color,
                style=style,
                label=edge_label if edge_color == "#2ECC71" else "",
                penwidth="1.5" if edge_color == "#2ECC71" else "1.0",
            )

    # Align non-migrated nodes of same generation
    for gen, eids in sorted(generations.items()):
        with g.subgraph() as s:
            s.attr(rank="same")
            for eid in eids:
                s.node(eid)

    # Legend
    with g.subgraph(name="cluster_legend") as legend:
        legend.attr(
            label="Legend",
            style="rounded",
            color="#CCCCCC",
            fontsize="10",
        )
        legend.node("leg_best", label="Best overall", fillcolor=BEST_COLOR, shape="box",
                     style="filled", color=ISLAND_COLORS[0], penwidth="2.5")
        legend.node("leg_i0", label="Island 0", fillcolor=ISLAND_LIGHT[0], shape="box",
                     style="filled", color=ISLAND_COLORS[0])
        legend.node("leg_i1", label="Island 1", fillcolor=ISLAND_LIGHT[1], shape="box",
                     style="filled", color=ISLAND_COLORS[1])
        legend.node("leg_mig", label="Migrated", fillcolor=ISLAND_LIGHT[0], shape="box",
                     style="filled", color=MIGRATED_BORDER, penwidth="2.0")
        legend.edge("leg_best", "leg_i0", style="invis")
        legend.edge("leg_i0", "leg_i1", style="invis")
        legend.edge("leg_i1", "leg_mig", style="invis")

    return g


def main():
    parser = argparse.ArgumentParser(description="Visualize evo-db lineage tree",
        usage="uv run visualize_lineage_tree.py <command> [options]",
    )
    parser.add_argument("db_files", nargs="+", help="Path to evo_db JSON file(s)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (without extension)")
    parser.add_argument("--format", "-f", default="png",
                        choices=["png", "svg", "pdf"],
                        help="Output format (default: png)")
    parser.add_argument("--view", action="store_true",
                        help="Open the rendered file after generation")
    args = parser.parse_args()

    for db_path in args.db_files:
        if not os.path.exists(db_path):
            print(f"ERROR: {db_path} not found", file=sys.stderr)
            sys.exit(1)

        db = load_db(db_path)
        name = os.path.splitext(os.path.basename(db_path))[0]
        output = args.output or name

        g = build_graph(db, name, fmt=args.format)
        out_path = g.render(output, cleanup=True, view=args.view)
        print(f"Rendered: {out_path}")


if __name__ == "__main__":
    main()
