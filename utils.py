import re


def parse_drill_file(file_path):
    """
    Parse a drill file and return the tools and coordinates.

    Returns:
    - A dict of the tool name: tool size of plated tools
    - A dict of the tool name: tool size of non-plated tools
    - A dict of the tool name: list of coordinates
    """
    plated_tools: dict[int, float] = {}
    non_plated_tools: dict[int, float] = {}
    drill_coordinates: dict[int, list[tuple[float, float]]] = {}
    current_tool = None
    is_plated = True  # Assume plated by default

    print(f"Parsing drill file: {file_path}")

    with open(file_path, "r") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if line.startswith("; #@! TA.AperFunction,"):
                is_plated = "Plated" in line
                print(f"Line {line_num}: Set is_plated to {is_plated}")
            elif line.startswith("T") and "C" in line:
                tool_match = re.match(r"T(\d+)C([\d.]+)", line)
                if tool_match:
                    tool_num = int(tool_match.group(1))
                    tool_size = float(tool_match.group(2))
                    if is_plated:
                        plated_tools[tool_num] = tool_size
                    else:
                        non_plated_tools[tool_num] = tool_size
                    drill_coordinates[tool_num] = []
                    print(
                        f"Line {line_num}: Added tool T{tool_num} (size: {tool_size}mm) to {'plated' if is_plated else 'non-plated'} tools"
                    )
            elif line.startswith("X") and "Y" in line:
                coords = re.findall(r"[XY]([-\d.]+)", line)
                if len(coords) == 2 and current_tool is not None:
                    x, y = map(float, coords)
                    drill_coordinates[current_tool].append((x, y))
                    print(
                        f"Line {line_num}: Added coordinates ({x}, {y}) to tool T{current_tool}"
                    )
            elif line.startswith("T"):
                tool_match = re.match(r"T(\d+)", line)
                if tool_match:
                    current_tool = int(tool_match.group(1))
                    print(f"Line {line_num}: Set current tool to T{current_tool}")

    print("\nParsing results:")
    print(f"Plated tools: {plated_tools}")
    print(f"Non-plated tools: {non_plated_tools}")
    for tool, coords in drill_coordinates.items():
        print(f"T{tool}: {len(coords)} holes")

    return plated_tools, non_plated_tools, drill_coordinates
