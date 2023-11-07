# Vesuvius Volume Browser

A simple browser built with Rust and [egui](https://github.com/emilk/egui) to browse volume data from the [Vesuvius Challenge](https://scrollprize.org/data) data set.

![demo](https://github.com/jrudolph/vesuvius-gui/assets/9868/261dfc1c-f9d5-41a4-8324-8963eef2afa2)

## Usage

Install required X11 libraries:
  * Ubuntu: `apt install -y libgl1 libxrandr2 libxi6 libxcursor1`

Download any number of cells from a [volume_grids folder](http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volume_grids/20230205180739/)
and put them into a directory. Then run the app with `./vesuvius-gui <path-to-directory>` / `cargo run --release <path-to-directory>`. The app will then
pick up whatever cells is there and display them.

## Data License

Accessing the data on https://vesuvius.virtual-void.net/ or through this app requires you to fill out the official
form and agree to the terms of the data license. See https://scrollprize.org/data.

## License

Code released under the MPL 2.0. See [LICENSE](LICENSE) for the full license text.
