from pathlib import Path

import cv2
import numpy as np
import rich
import typer
from PIL import Image
from rich.progress import track
from rich.prompt import Prompt
from rich.table import Table
from scipy.interpolate import Rbf

import calibration_package
import config

my_dir = Path(__file__).parent
build_dir = my_dir / "build"
calibration_dir = my_dir / "calibrations"
calibration_cache = calibration_dir / ".cache"


def make_remap_mapping(remapper: tuple[Rbf, Rbf], image_size: tuple[int, int], px_per_mm: float, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Find the smallest dtype that can hold the remapped integer values
    # dtype = np.min_scalar_type(max(input_grid_width, input_grid_height))
    dtype = np.float32  # Remap demands float32

    image_width, image_height = image_size

    total_points = image_height * image_width
    map_inverse_x = np.zeros((total_points,), dtype=dtype)
    map_inverse_y = np.zeros((total_points,), dtype=dtype)

    # Create full grid coordinates
    grid_x, grid_y = np.meshgrid(
        np.arange(image_width),
        np.arange(image_height),
        indexing='xy'
    )

    # Convert to mm coordinates
    grid_points_x = grid_x.ravel() / px_per_mm
    grid_points_y = grid_y.ravel() / px_per_mm

    # Process in chunks
    for chunk_start in track(range(0, total_points, chunk_size), description="Processing remapping array..."):
        chunk_end = min(chunk_start + chunk_size, total_points)

        # Extract current chunk
        grid_points_x_chunk = grid_points_x[chunk_start:chunk_end]
        grid_points_y_chunk = grid_points_y[chunk_start:chunk_end]

        # Apply RBF interpolation
        map_inverse_chunk_x = remapper[0](grid_points_x_chunk, grid_points_y_chunk) * px_per_mm
        map_inverse_chunk_y = remapper[1](grid_points_x_chunk, grid_points_y_chunk) * px_per_mm

        # Clip to valid pixel coordinates
        map_inverse_chunk_x = np.clip(map_inverse_chunk_x, 0, image_width - 1)
        map_inverse_chunk_y = np.clip(map_inverse_chunk_y, 0, image_height - 1)

        # Store results
        map_inverse_x[chunk_start:chunk_end] = map_inverse_chunk_x.astype(dtype)
        map_inverse_y[chunk_start:chunk_end] = map_inverse_chunk_y.astype(dtype)

    # Reshape back to grid format
    map_inverse_x = map_inverse_x.reshape(image_height, image_width)
    map_inverse_y = map_inverse_y.reshape(image_height, image_width)

    return map_inverse_x, map_inverse_y


def remap_image(image: np.ndarray, remapper: tuple[np.ndarray, np.ndarray], output_size_px: tuple[int, int]) -> np.ndarray:
    map_inverse_x, map_inverse_y = remapper

    restored_image = cv2.remap(
        image,
        map_inverse_x,
        map_inverse_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    width_start = (restored_image.shape[1] - output_size_px[0]) // 2
    height_start = (restored_image.shape[0] - output_size_px[1]) // 2
    restored_image = restored_image[height_start:height_start+output_size_px[1], width_start:width_start+output_size_px[0]]

    return restored_image


def get_image_dpi(image_path: Path) -> float | None:
    Image.MAX_IMAGE_PIXELS = None  # Disable the PIL limit because we own everything
    info = Image.open(image_path).info
    if "dpi" not in info:
        return None
    return info["dpi"][0]


def main(
    targets: list[Path] = typer.Argument(..., help="Paths to the target images to remap"),
    calibration_path: Path | None = typer.Option(None, help="Path to the calibration file"),
    output_path: Path | None = typer.Option(None, help="Path to the output directory"),
    output_suffix: str = typer.Option("restored", help="Suffix to add to the output files"),
    remap_chunk_size: int = typer.Option(10000, help="Chunk size for remap mapping"),
    input_dpi: float | None = typer.Option(None, help="Input DPI of the target images"),
    debug: bool = typer.Option(False, help="Show debuggy things", flag_value=True),
):
    if not targets:
        target_choices = list(Path(".").glob("*.png"))
        if not target_choices:
            rich.print("[red]No PNG files found in current directory[/red]")
            raise typer.Exit(1)

        targets = Prompt.ask(
            "Select target images to process",
            choices=[str(p) for p in target_choices],
            show_choices=True,
            multiple=True
        )
        targets = [Path(t) for t in targets]

    if calibration_path is None:
        def _try_load_calibration(path: Path) -> calibration_package.CalibrationPackage | None:
            try:
                return calibration_package.CalibrationPackage.load(path)
            except Exception as e:
                rich.print(f"[red]Error loading calibration file {path}: {e}[/red]")
                return None

        calibration_choices = {}
        calibration_paths = {}
        option = 1
        table = Table("Option", "Filename", "Description", title="Calibration Files")
        for path in calibration_dir.glob("*.pkl"):
            if cal_pkg := _try_load_calibration(path):
                table.add_row(str(option), path.name, cal_pkg.description)
                calibration_choices[option] = cal_pkg
                calibration_paths[option] = path
                option += 1

        if not calibration_choices:
            rich.print(f"[red]No calibration files found in {calibration_dir} directory[/red]")
            raise typer.Exit(1)

        if len(calibration_choices) == 1:
            rich.print("[yellow]Only one calibration file found, using it by default[/yellow]")
            calibration_selected = 1
        else:
            rich.print(table)

            calibration_selected = int(Prompt.ask(
                "Select calibration file",
                choices=[str(k) for k in calibration_choices.keys()],
                show_choices=True
            ))

        cal_pkg = calibration_choices[calibration_selected]
        calibration_path = calibration_paths[calibration_selected]
    else:
        cal_pkg = calibration_package.CalibrationPackage.load(calibration_path)

    if input_dpi is None:
        input_dpi = get_image_dpi(targets[0])

    if input_dpi is None:
        rich.print("[red]Could not determine DPI of target images[/red]")
        raise typer.Exit(1)

    target_px_per_mm = input_dpi / 25.4

    input_width_mm = cal_pkg.bounds_mm[1][0] - cal_pkg.bounds_mm[0][0]
    input_height_mm = cal_pkg.bounds_mm[1][1] - cal_pkg.bounds_mm[0][1]

    input_width_px = int(input_width_mm * target_px_per_mm)
    input_height_px = int(input_height_mm * target_px_per_mm)

    # Check the calibration cache for a saved npy remapping
    calibration_cache.mkdir(parents=True, exist_ok=True)
    remap_path_suffix = calibration_cache / f"{calibration_path.stem}-{round(target_px_per_mm, 2)}"
    if remap_path_suffix.with_suffix(".x.npy").exists() and remap_path_suffix.with_suffix(".y.npy").exists():
        rich.print("[green]Loading remap from cache[/green]")
        remap_maps = (
            np.load(remap_path_suffix.with_suffix(".x.npy")),
            np.load(remap_path_suffix.with_suffix(".y.npy"))
        )
    else:
        rich.print("[green]Creating remap[/green]")
        remap_maps = make_remap_mapping(cal_pkg.remap_mm_mapper, (input_width_px, input_height_px), target_px_per_mm, remap_chunk_size)
        rich.print("[green]Saving remap to cache[/green]")
        np.save(remap_path_suffix.with_suffix(".x.npy"), remap_maps[0])
        np.save(remap_path_suffix.with_suffix(".y.npy"), remap_maps[1])

    for target_image_path in track(targets, description="Processing targets..."):
        rich.print(f"[green]Processing {target_image_path}[/green]")
        target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
        target_image = cv2.resize(target_image, (input_width_px, input_height_px), interpolation=cv2.INTER_AREA)
        restored_image = remap_image(target_image, remap_maps, config.config["output_size"], debug)

        if output_path is None:
            image_output_path = target_image_path.with_suffix(f"{output_suffix}.png")
        else:
            image_output_path = output_path / (target_image_path.stem + f"{output_suffix}.png")

        rich.print(f"[green]Writing {image_output_path}[/green]")
        cv2.imwrite(image_output_path, restored_image)

        if debug:
            # Display the restored image as blue
            # and the original image as red
            display_image = np.zeros((restored_image.shape[0], restored_image.shape[1], 3), dtype=np.uint8)
            display_image[:, :, 0] = cv2.bitwise_not(restored_image)
            display_image[:, :, 1] = cv2.bitwise_not(target_image)
            display_image[:, :, 2] = 0
            debug_image_output_path = image_output_path.with_suffix(".debug.png")
            rich.print(f"[green]Writing {debug_image_output_path}[/green]")
            cv2.imwrite(debug_image_output_path, display_image)

    rich.print("[green]Done![/green]")

if __name__ == "__main__":
    typer.run(main)
