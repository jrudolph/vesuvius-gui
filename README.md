# Vesuvius Volume Browser

A simple browser built with Rust and [egui](https://github.com/emilk/egui) to browse volume data from the [Vesuvius Challenge](https://scrollprize.org/data) data set.

![demo](media/v26-zoomed-out-segment.jpg)

It does not require to download any data upfront.

All the published volumes published so far are supported:

- Scroll 1
- Scroll 2
- Scroll 0332
- Scroll 1667
- Scroll 172
- Fragments 1 - 4 (from the Kaggle competition)
- Fragment PHerc0051Cr04Fr08
- Fragment PHerc1667Cr01Fr03

Known surface segments are shown in the catalog and can be rendered on a 4th pane.

## Features

- Access to the full volume data set, data is converted to a more efficient format by a background server
- A catalog of known surface segments will allow on-the-fly downloading of surfaces meshes and live rendering
- Rendering options for volumes:
  - thresholding
  - bit depth reduction
  - showing different bit planes
- Rendering options for surfaces:
  - trilinear interpolation
  - show surface outline on the volume panes
  - show xyz outline on the surface pane
  - synchronized panning and zooming between the panes

## Installation

Grab a binary from [the latest release](https://github.com/jrudolph/vesuvius-gui/releases).

## Usage

Install required X11 libraries:

- Ubuntu: `apt install -y libgl1 libxrandr2 libxi6 libxcursor1`
- MacOSX: Should work out of the box
- Windows: Should work out of the box

### Simple browsing:

```
./vesuvius-gui
```

When run without any arguments, the app will download the volume data from the tiles server and cache them in a local directory (below the OS-specific cache directory).

### Specifying options:

Use the `--help` flag to see all available options:

```
Vesuvius GUI, an app to visualize and explore 3D data of the Vesuvius Challenge (https://scrollprize.org)

Usage: vesuvius-gui [OPTIONS]

Options:
  -d, --data-directory <DATA_DIRECTORY>
          Override the data directory. By default, a directory in the user's cache is used
      --obj <OBJ>
          Browse segment from obj file. You need to also provide --width and --height. Provide the --volume if the segment does not target Scroll 1a / 20230205180739
      --width <WIDTH>
          Width of the segment file when browsing obj files
      --height <HEIGHT>
          Height of the segment file when browsing obj files
  -o, --overlay <OVERLAY>
          A directory that contains data to overlay. Only zarr arrays are currently supported
  -v, --volume [<VOLUME>]
          The id of a volume to open
  -h, --help
          Print help
```

## Data License

Accessing the data on https://vesuvius.virtual-void.net/ or through this app requires you to fill out the official
form and agree to the terms of the data license. See https://scrollprize.org/data.

## License

Code released under the MPL 2.0. See [LICENSE](LICENSE) for the full license text.
