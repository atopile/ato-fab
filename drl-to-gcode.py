import re
import json
import os

# Global settings
RETRACT_HEIGHT = 5  # Global retract height in mm

# Parameters
BOARD_THICKNESS = 1.6  # Default board thickness in mm
RAPID_Z = 2    # Height for rapid movements between holes
APPROACH_Z = .5  # Height from which to start feeding into material

def parse_drill_file(file_path):
    plated_tools = {}
    non_plated_tools = {}
    drill_coordinates = {}
    current_tool = None
    is_plated = True  # Assume plated by default

    print(f"Parsing drill file: {file_path}")

    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line.startswith('; #@! TA.AperFunction,'):
                is_plated = 'Plated' in line
                print(f"Line {line_num}: Set is_plated to {is_plated}")
            elif line.startswith('T') and 'C' in line:
                tool_match = re.match(r'T(\d+)C([\d.]+)', line)
                if tool_match:
                    tool_num = int(tool_match.group(1))
                    tool_size = float(tool_match.group(2))
                    if is_plated:
                        plated_tools[tool_num] = tool_size
                    else:
                        non_plated_tools[tool_num] = tool_size
                    drill_coordinates[tool_num] = []
                    print(f"Line {line_num}: Added tool T{tool_num} (size: {tool_size}mm) to {'plated' if is_plated else 'non-plated'} tools")
            elif line.startswith('X') and 'Y' in line:
                coords = re.findall(r'[XY]([-\d.]+)', line)
                if len(coords) == 2 and current_tool is not None:
                    x, y = map(float, coords)
                    drill_coordinates[current_tool].append((x, y))
                    print(f"Line {line_num}: Added coordinates ({x}, {y}) to tool T{current_tool}")
            elif line.startswith('T'):
                tool_match = re.match(r'T(\d+)', line)
                if tool_match:
                    current_tool = int(tool_match.group(1))
                    print(f"Line {line_num}: Set current tool to T{current_tool}")

    print("\nParsing results:")
    print(f"Plated tools: {plated_tools}")
    print(f"Non-plated tools: {non_plated_tools}")
    for tool, coords in drill_coordinates.items():
        print(f"T{tool}: {len(coords)} holes")

    return plated_tools, non_plated_tools, drill_coordinates

def load_tool_library(file_path):
    with open(file_path, 'r') as file:
        library = json.load(file)
    
    # Check if each tool has the required 'depth_offset'
    for tool in library['drills']:
        if 'depth_offset' not in tool:
            raise KeyError(f"'depth_offset' is missing for tool: {tool['name']}")
    
    return library

def find_matching_tool(target_diameter, tolerance, tool_library):
    for tool in tool_library['drills']:
        if abs(tool['diameter'] - target_diameter) <= tolerance:
            return tool
    return None

def generate_gcode(output_file, matched_tools, drill_coordinates, board_thickness=BOARD_THICKNESS):
    if not drill_coordinates:
        print(f"No holes to process for {output_file}. Skipping G-code generation.")
        return

    all_coordinates = [coord for coords in drill_coordinates.values() for coord in coords]

    min_x = min(coord[0] for coord in all_coordinates)
    min_y = min(coord[1] for coord in all_coordinates)

    # Add a -10mm offset to both X and Y
    offset_x = min_x - 10
    offset_y = min_y - 10

    with open(output_file, 'w') as outfile:
        outfile.write("(fusion-gcode-drill)\n")
        outfile.write("G90 G21\n\n")  # Absolute mode, Millimeter mode

        for drill_tool_num, coordinates in drill_coordinates.items():
            library_tool = matched_tools.get(drill_tool_num)
            if library_tool:
                tool_number = library_tool['number']
                drill_depth = board_thickness + library_tool['depth_offset']
                outfile.write(f"(T{tool_number}  D={library_tool['diameter']} CR=0 TAPER=118deg - ZMIN=-{drill_depth:.3f} - drill)\n")
                outfile.write(f"M6 T{tool_number}\n")  # Auto tool change
                outfile.write(f"M3 S{library_tool['spindle_speed']}\n")  # Start spindle
                outfile.write("M801 S100\n")  # Turn on vacuum at full power

                outfile.write(f"G0 Z{RAPID_Z} F3000\n")  # Rapid move to safe Z height

                for x, y in coordinates:
                    outfile.write(f"G0 X{x - offset_x:.3f} Y{y - offset_y:.3f}\n")
                    outfile.write(f"G0 Z{APPROACH_Z:.3f}\n")
                    outfile.write(f"G1 Z-{drill_depth:.3f} F{library_tool['plunge_rate']}\n")
                    outfile.write(f"G0 Z{RAPID_Z:.3f}\n")

                outfile.write("M5\n")  # Stop spindle
                outfile.write("M802\n")  # Turn off vacuum
            else:
                print(f"Warning: No matching tool found for drill size {drill_tool_num}. Skipping these holes.")

        outfile.write("G28\n")  # Go to clearance position
        outfile.write("M30\n")  # End of program

# Create output directory if it doesn't exist
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")


# Usage
input_file = 'default.drl'
pth_output_file = 'outputs/drill-pth.cnc'
npth_output_file = 'outputs/drill-npth.cnc'
tool_library_file = 'tool_library.json'

try:
    tool_library = load_tool_library(tool_library_file)
    plated_tools, non_plated_tools, drill_coordinates = parse_drill_file(input_file)
    tolerance = 0.05

    print("Plated tools:", plated_tools)
    print("Non-plated tools:", non_plated_tools)
    print("Tool library:", json.dumps(tool_library, indent=2))

    # Match drill tools
    matched_pth_tools = {}
    matched_npth_tools = {}
    for tool_num, size in plated_tools.items():
        matched_tool = find_matching_tool(size, tolerance, tool_library)
        if matched_tool:
            matched_pth_tools[tool_num] = matched_tool
        else:
            print(f"Warning: No matching tool found for plated hole size {size}mm")

    for tool_num, size in non_plated_tools.items():
        matched_tool = find_matching_tool(size, tolerance, tool_library)
        if matched_tool:
            matched_npth_tools[tool_num] = matched_tool
        else:
            print(f"Warning: No matching tool found for non-plated hole size {size}mm")

    # Generate G-code for plated holes
    if matched_pth_tools:
        generate_gcode(pth_output_file, matched_pth_tools, {k: v for k, v in drill_coordinates.items() if k in plated_tools})
    else:
        print("No plated holes to process. Skipping PTH G-code generation.")

    # Generate G-code for non-plated holes
    if matched_npth_tools:
        generate_gcode(npth_output_file, matched_npth_tools, {k: v for k, v in drill_coordinates.items() if k in non_plated_tools})
    else:
        print("No non-plated holes to process. Skipping NPTH G-code generation.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
