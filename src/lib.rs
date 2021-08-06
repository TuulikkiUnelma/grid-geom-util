mod line;
mod point;
mod rect;

pub use line::*;
pub use point::*;
pub use rect::*;

use std::fmt;

/// A generic max function.
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

/// A generic min function.
fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

/// A cardinal direction towards one of the axises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CardDir {
    /// Towards positive X
    East,
    /// Towards positive Y
    South,
    /// Towards negative X
    West,
    /// Towards negative Y
    North,
}

impl fmt::Display for CardDir {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CardDir::*;
        f.write_str(match self {
            East => "east",
            South => "south",
            West => "west",
            North => "north",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_points(min: i32, max: i32) -> impl Iterator<Item = Point<i32>> {
        let mut x = min;
        let mut y = min;
        std::iter::from_fn(move || {
            let out = Point { x, y };
            if x < max {
                x += 1;
            } else if y < max {
                x = min;
                y += 1;
            } else {
                return None;
            }
            Some(out)
        })
    }

    #[test]
    fn point_distance() {
        for p in test_points(-500, 500) {
            let (x, y) = p.into();

            let eucl = p.distance_euclid();
            let taxi = p.distance_taxi();
            let king = p.distance_king();

            assert_eq!(eucl, ((x * x + y * y) as f32).sqrt());
            assert_eq!(taxi, x.abs() + y.abs());
            assert_eq!(king, x.abs().max(y.abs()));
        }
    }

    #[test]
    fn point_cardinal_dir() {
        assert_eq!(Point::new(5, 1).cardinal(), CardDir::East);
        assert_eq!(Point::new(-5, 1).cardinal(), CardDir::West);
        assert_eq!(Point::new(1, -5).cardinal(), CardDir::North);
        assert_eq!(Point::new(1, 5).cardinal(), CardDir::South);
    }
}
