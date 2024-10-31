"""
Script to generate SVG files required to manufacture
PCBs using both laser drilling and structuring.
"""

# %%

import logging
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import rich
import semver
import svg
import svgpathtools
from kicadcliwrapper.generated.kicad_cli import kicad_cli
from rich.logging import RichHandler
from rich.table import Table
from ruamel.yaml import YAML
from shapely.geometry import Polygon
from shapely.ops import unary_union
import svgpathtools
import numpy as np


# import utils

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)


# class Checker:
#     _checks: list["Check"]
#     def __init__(self):
#         self._checks = []

#     @dataclass
#     class Check:
#         name: str
#         description: str
#         result: bool

#     def report(self):
#         table = Table("Name", "Description", "Result", title="Checks")

#         for check in self._checks:
#             table.add_row(check.name, check.description, "✅" if check.result else "❌")

#         rich.print(table)

#     def check(self, func):
#         check = self.Check(name=func.__name__, description=func.__doc__, result=True)
#         try:
#             return func()
#         except Exception:
#             check.result = False
#             self.report()
#             exit(1)  # TODO: allow bundling of errors
#         finally:
#             self._checks.append(check)


# checker = Checker()


# Load config
# yaml = YAML()
# if config_files := list(Path(__name__).parent.glob("config.y*ml")):
#     config = yaml.load(config_files[0])
# else:
#     config = {}

# Ensure output directory exists
# build_dir = Path("build")
# build_dir.mkdir(exist_ok=True)

# TODO: wrap in a name == __main__ guard
# TODO: switch back
# inputs_dir = Path(".")  # default to current directory
# inputs_dir = Path("robo-plater/elec/layout/plating-bath-power-supply-demo")  # default to current directory

def find_one_file(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError(
            "Expected exactly one file matching"
            f" pattern {pattern}, but found {len(files)}"
)
    return files[0]

# input_pcb_file = find_one_file(inputs_dir, "*.kicad_pcb")

# Check KiCAD version and make sure it's working
if version := kicad_cli(kicad_cli.version()).exec():
    log.info("Found KiCAD version: %s", version)
    min_version = "8.0.6"
    if semver.compare(version, min_version) < 0:
        log.error("KiCAD version must be >=%s", min_version)
        exit(1)

# Generate KiCAD exports
# kicad_build_dir = build_dir / "kicad"
# kicad_build_dir.mkdir(exist_ok=True)

# TODO: move to utils file
def kicad_export(command) -> str:
    return kicad_cli(
        kicad_cli.pcb(
            kicad_cli.pcb.export(
                command
            )
        ),
    ).exec(check=True)

# kicad_export(
#     kicad_cli.pcb.export.drill(
#         INPUT_FILE=str(input_pcb_file.absolute()),
#         output=str(kicad_build_dir.absolute()),
#     )
# )

# TODO: move to utils file
def export_svg(input_file: Path, layer: str, output_dir: Path) -> Path:
    output_path = (output_dir / f"{layer}.svg").absolute()
    kicad_export(
        kicad_cli.pcb.export.svg(
            INPUT_FILE=str(input_file.absolute()),
            output=str(output_path),
            black_and_white=True,
            layers=layer,
            exclude_drawing_sheet=True,
        )
    )
    return output_path


def export_dxf(input_file: Path, layer: str, output_dir: Path) -> Path:
    output_path = (output_dir / f"{layer}.dxf").absolute()
    kicad_export(
        kicad_cli.pcb.export.dxf(
            INPUT_FILE=str(input_file.absolute()),
            output=str(output_path),
            layers=layer,
            output_units="mm",
            include_border_title=False,
            exclude_refdes=True,
            exclude_value=True,
        )
    )
    return output_path

# edge_cuts_file = export_svg(input_pcb_file, "Edge.Cuts", kicad_build_dir)

# Find bounding box
# paths, _ = svgpathtools.svg2paths(edge_cuts_file)
def paths_bounding_box(paths: list[svgpathtools.Path]):
    # Create an empty list to store all bounding boxes
    bboxes = []

    # Collect bounding boxes for all paths
    for path in paths:
        bboxes.append(path.bbox())

    # Convert to NumPy array for efficient computation
    bboxes = np.array(bboxes)

    # Compute the overall bounding box
    xmin, xmax = np.min(bboxes[:, [0, 1]]), np.max(bboxes[:, [0, 1]])
    ymin, ymax = np.min(bboxes[:, [2, 3]]), np.max(bboxes[:, [2, 3]])

    return xmin, xmax, ymin, ymax


# xmin, xmax, ymin, ymax = paths_bounding_box(paths)
# log.info(f"Bounding box: {xmin=}, {xmax=}, {ymin=}, {ymax=}")

# @checker.check
# def check_bounding_box():
#     if working_area := config.get("working_area"):
#         assert isinstance(working_area, dict)
#         x_span = xmax - xmin
#         y_span = ymax - ymin

#         if x_area := working_area.get("x"):
#             assert x_span <= x_area

#         if y_area := working_area.get("y"):
#             assert y_span <= y_area


# origin_offset = (xmin, ymin)
# board_width = xmax - xmin
# board_height = ymax - ymin


# Generate holes SVG files for laser drilling
# drill_file = find_one_file(kicad_build_dir, "*.drl")
# plated_tools, unplated_tools, holes = utils.parse_drill_file(drill_file)

# def generate_holes_svg(
#     holes: dict[int, list[tuple[float, float]]],
#     tools: dict[int, float],
#     hole_color: str,
#     output_file: Path,
# ) -> svg.SVG:
#     def map_holes(holes: dict[int, list[tuple[float, float]]], tools: dict[int, float]):
#         output = []
#         for tool, dia in tools.items():
#             for pos in holes[tool]:
#                 output.append((dia, pos))
#         return output

#     target_list = map_holes(holes, tools)
#     circles = []

#     for dia, (x, y) in target_list:
#         hole = svg.Circle(
#             cx = x - origin_offset[0],
#             cy = -y - origin_offset[1],
#             r = dia/2,
#             fill = hole_color,
#         )
#         circles.append(hole)

#     _svg = svg.SVG(
#         width=board_width,
#         height=board_height,
#         elements=circles,
#     )

#     output_file.write_text(str(_svg))

# generate_holes_svg(holes, plated_tools, "red", build_dir / "plated_holes.svg")
# generate_holes_svg(holes, unplated_tools, "blue", build_dir / "unplated_holes.svg")
# # %%
# # Process SVG outlines for material to be removed
# top_copper_file = export_svg(input_pcb_file, "F.Cu", kicad_build_dir)

# # %%
# paths, attributes = svgpathtools.svg2paths(top_copper_file)

# def svgpathtools_to_shapely(
#     paths: list[svgpathtools.Path],
#     tolerance: float = 1,
# ) -> list[Polygon]:
#     polygons = []
#     for path_i, path in enumerate(paths):
#         polygon_points = []
#         def _dedup(point):
#             if polygon_points and polygon_points[-1] == point:
#                 return
#             polygon_points.append(point)

#         for section_i, section in enumerate(path):
#             if isinstance(section, svgpathtools.Line):
#                 _dedup(section.start)
#                 polygon_points.append(section.end)
#             else:
#                 steps = math.ceil(path.length() / tolerance)
#                 _dedup(path.start)
#                 for T in np.linspace(1/steps, 1, steps):
#                     polygon_points.append(path.point(T))

#         polygon_points = [(p.real, p.imag) for p in polygon_points]
#         if len(polygon_points) < 3:
#             log.warning(f"Skipping path {path_i} with less than 3 points")
#             continue

#         polygon = Polygon(polygon_points)
#         polygons.append(polygon)
#     return polygons

# shapely_polys = svgpathtools_to_shapely(paths)

# union = unary_union(shapely_polys)

# # Save the result
# svgpathtools.wsvg(union, filename='union_output.svg')

# # export_svg(input_pcb_file, "B.Cu", kicad_build_dir)

# # %%
# top_copper_file = Path("build/kicad/F.Cu.svg").absolute()

# Report results
# checker.report()
# log.info("Done! :sparkles:")
