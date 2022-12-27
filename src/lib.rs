#![doc = include_str!("../README.md")]

mod circle;
mod line;
mod point;
mod rect;

pub use circle::*;
pub use line::*;
pub use point::*;
pub use rect::*;

use std::fmt;

/// A generic max function.
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}

/// A generic min function.
fn min<T: PartialOrd>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

/// A cardinal direction towards one of the axises.
///
/// Returned by [`Point::cardinal`].
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
    use itertools::Itertools;
    use std::collections::HashSet;

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

            let euclid = p.map(|x| x as f32).distance_euclid();
            let taxi = p.distance_taxi();
            let king = p.distance_king();

            assert_eq!(euclid, ((x * x + y * y) as f32).sqrt());
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

    #[test]
    fn point_neighborhood() {
        let f = Point::new;
        let p = f(10, -15);

        for (i, (n, m)) in p
            .neighbors_neumann_iter()
            .zip(p.neighbors_moore_iter())
            .enumerate()
        {
            // first 4 should be same
            assert_eq!(n, m);
            assert!(i <= 4);
        }

        let neumann: HashSet<Point<i32>> = vec![f(11, -15), f(10, -16), f(9, -15), f(10, -14)]
            .into_iter()
            .collect();

        let diag: HashSet<Point<i32>> = vec![f(11, -16), f(9, -16), f(9, -14), f(11, -14)]
            .into_iter()
            .collect();
        let moore: HashSet<Point<i32>> = diag.union(&neumann).copied().collect();

        assert_eq!(neumann, p.neighbors_neumann_iter().collect());
        assert_eq!(moore, p.neighbors_moore_iter().collect());
    }

    #[test]
    fn line_lerp() {
        let line = Line::new(5, -5, 15, -25);

        let f = Point::new;
        let g = |t: f32| line.lerp(t).map(|x| x as i32);

        assert_eq!(f(5, -5), g(0.0));
        assert_eq!(f(6, -7), g(0.1));
        assert_eq!(f(7, -9), g(0.2));
        assert_eq!(f(8, -11), g(0.3));
        assert_eq!(f(9, -13), g(0.4));
        assert_eq!(f(10, -15), g(0.5));
        assert_eq!(f(11, -17), g(0.6));
        assert_eq!(f(12, -19), g(0.7));
        assert_eq!(f(13, -21), g(0.8));
        assert_eq!(f(14, -23), g(0.9));
        assert_eq!(f(15, -25), g(1.0));
    }

    #[test]
    fn line_bresenham() {
        let line = Line::new(5, -4, 7, -12);
        let f = Point::new;

        let points: HashSet<Point<i32>> = vec![
            f(5, -4),
            f(5, -5),
            f(5, -6),
            f(6, -7),
            f(6, -8),
            f(6, -9),
            f(6, -10),
            f(7, -11),
            f(7, -12),
        ]
        .into_iter()
        .collect();

        assert_eq!(points, line.bresenham_iter().collect());
    }

    #[test]
    fn rect_new_check() {
        let f = Point::new;

        for (p, delta) in test_points(-50, 50).zip(
            [
                f(0, 0),
                f(0, 2),
                f(1, 1),
                f(2, 1),
                f(2, 2),
                f(3, 3),
                f(4, 5),
            ]
            .iter()
            .cycle(),
        ) {
            let (x1, y1, x2, y2) = (p.x, p.y, p.x + delta.x, p.y + delta.y);
            let (xm2, ym2) = (p.x - delta.x, p.y - delta.y);
            unsafe {
                // normal
                assert_eq!(
                    Rect::new(x1, y1, x2, y2),
                    Rect::new_unchecked(x1, y1, x2, y2)
                );

                // x flipped
                assert_eq!(
                    Rect::new(x1, y1, xm2, y2),
                    Rect::new_unchecked(xm2, y1, x1, y2)
                );

                // y flipped
                assert_eq!(
                    Rect::new(x1, y1, x2, ym2),
                    Rect::new_unchecked(x1, ym2, x2, y1)
                );

                // both flipped
                assert_eq!(
                    Rect::new(x1, y1, xm2, ym2),
                    Rect::new_unchecked(xm2, ym2, x1, y1)
                );
            }
        }
    }

    #[test]
    fn rect_point_iters_1x1() {
        let f = |x, y| Some(Point::new(x, y));

        let rect_1x1 = Rect::new(6, 6, 6, 6);
        let mut p = rect_1x1.points_iter();
        let mut p_b = rect_1x1.points_border_iter();

        assert_eq!(p.next(), f(6, 6));
        assert_eq!(p.next(), None);

        assert_eq!(p_b.next(), f(6, 6));
        assert_eq!(p_b.next(), None);
    }

    #[test]
    fn rect_point_iters_4x1() {
        let f = |x, y| Some(Point::new(x, y));

        let rect_3x1 = Rect::new(5, 5, 8, 5);
        let mut p = rect_3x1.points_iter();
        let mut p_b = rect_3x1.points_border_iter();

        assert_eq!(p.next(), f(5, 5));
        assert_eq!(p.next(), f(6, 5));
        assert_eq!(p.next(), f(7, 5));
        assert_eq!(p.next(), f(8, 5));
        assert_eq!(p.next(), None);

        assert_eq!(p_b.next(), f(5, 5));
        assert_eq!(p_b.next(), f(6, 5));
        assert_eq!(p_b.next(), f(7, 5));
        assert_eq!(p_b.next(), f(8, 5));
        assert_eq!(p_b.next(), None);
    }

    #[test]
    fn rect_point_iters_1x4() {
        let f = |x, y| Some(Point::new(x, y));

        let rect_1x3 = Rect::new(5, 5, 5, 8);
        let mut p = rect_1x3.points_iter();
        let mut p_b = rect_1x3.points_border_iter();

        assert_eq!(p.next(), f(5, 5));
        assert_eq!(p.next(), f(5, 6));
        assert_eq!(p.next(), f(5, 7));
        assert_eq!(p.next(), f(5, 8));
        assert_eq!(p.next(), None);

        assert_eq!(p_b.next(), f(5, 5));
        assert_eq!(p_b.next(), f(5, 6));
        assert_eq!(p_b.next(), f(5, 7));
        assert_eq!(p_b.next(), f(5, 8));
        assert_eq!(p_b.next(), None);
    }

    #[test]
    fn rect_point_iters_3x3() {
        let f = |x, y| Some(Point::new(x, y));

        let rect_3x3 = Rect::new(-1, -1, 1, 1);
        let mut p = rect_3x3.points_iter();
        let mut p_b = rect_3x3.points_border_iter();

        assert_eq!(p.next(), f(-1, -1));
        assert_eq!(p.next(), f(0, -1));
        assert_eq!(p.next(), f(1, -1));
        assert_eq!(p.next(), f(-1, 0));
        assert_eq!(p.next(), f(0, 0));
        assert_eq!(p.next(), f(1, 0));
        assert_eq!(p.next(), f(-1, 1));
        assert_eq!(p.next(), f(0, 1));
        assert_eq!(p.next(), f(1, 1));
        assert_eq!(p.next(), None);

        assert_eq!(p_b.next(), f(-1, -1));
        assert_eq!(p_b.next(), f(0, -1));
        assert_eq!(p_b.next(), f(1, -1));
        assert_eq!(p_b.next(), f(1, 0));
        assert_eq!(p_b.next(), f(1, 1));
        assert_eq!(p_b.next(), f(0, 1));
        assert_eq!(p_b.next(), f(-1, 1));
        assert_eq!(p_b.next(), f(-1, 0));
        assert_eq!(p_b.next(), None);
    }

    #[test]
    fn rect_shrink() {
        let f = Rect::new;

        let r1x1 = f(8, 8, 8, 8);
        let r4x1 = f(5, 5, 8, 5);
        let r1x4 = f(5, 5, 5, 8);
        let r4x4 = f(0, 0, 3, 3);

        assert_eq!(r1x1.shrink([10, 10]), r1x1);

        assert_eq!(r4x1.shrink([1, 0]), f(6, 5, 7, 5));
        assert_eq!(r4x1.shrink([2, 0]), f(6, 5, 6, 5));
        assert_eq!(r4x1.shrink([3, 0]), f(6, 5, 6, 5));
        assert_eq!(r4x1.shrink([0, 1]), r4x1);
        assert_eq!(r4x1.shrink([0, 10]), r4x1);

        assert_eq!(r1x4.shrink([0, 1]), f(5, 6, 5, 7));
        assert_eq!(r1x4.shrink([0, 2]), f(5, 6, 5, 6));
        assert_eq!(r1x4.shrink([0, 3]), f(5, 6, 5, 6));
        assert_eq!(r1x4.shrink([1, 0]), r1x4);
        assert_eq!(r1x4.shrink([10, 0]), r1x4);

        assert_eq!(r4x4.shrink([1, 1]), f(1, 1, 2, 2));
        assert_eq!(r4x4.shrink([2, 2]), f(1, 1, 1, 1));
        assert_eq!(r4x4.shrink([3, 3]), f(1, 1, 1, 1));

        assert_eq!(r4x4.shrink([1, 0]), f(1, 0, 2, 3));
        assert_eq!(r4x4.shrink([2, 0]), f(1, 0, 1, 3));
        assert_eq!(r4x4.shrink([3, 0]), f(1, 0, 1, 3));

        assert_eq!(r4x4.shrink([0, 1]), f(0, 1, 3, 2));
        assert_eq!(r4x4.shrink([0, 2]), f(0, 1, 3, 1));
        assert_eq!(r4x4.shrink([0, 3]), f(0, 1, 3, 1));
    }

    #[test]
    fn snap_points_unsigned() {
        let point_range = |x1, x2, y1, y2| {
            (y1..=y2)
                .cartesian_product(x1..=x2)
                .map(|(y, x)| Point { x, y })
        };

        for grid_incr in point_range(1u32, 10, 1, 10) {
            for (grid_point, local_point) in point_range(0u32, 10, 0, 10)
                .cartesian_product(point_range(0, grid_incr.x - 1, 0, grid_incr.y - 1))
            {
                let grid_point_pos = grid_point * grid_incr;
                if Point::op_any_many([local_point, grid_point_pos, grid_incr], |slice| {
                    if let [local, grid, incr] = *slice {
                        grid == 0 && local < incr / 2
                    } else {
                        unreachable!()
                    }
                }) {
                    // outside grid area, a local coordinate would be negative
                    continue;
                }
                let local_point_pos = local_point + grid_point_pos - grid_incr.map(|x| (x - 1) / 2);
                assert_eq!(grid_point_pos, local_point_pos.snap(&grid_incr));
            }
        }
    }

    #[test]
    fn snap_points_signed() {
        let point_range = |x1, x2, y1, y2| {
            (y1..=y2)
                .cartesian_product(x1..=x2)
                .map(|(y, x)| Point { x, y })
        };

        for grid_incr in point_range(1, 10, 1, 10) {
            let w = (grid_incr.x - 1) as f32;
            let h = (grid_incr.y - 1) as f32;
            let x_ext_neg = -((w / 2.0).floor() as i32);
            let x_ext_pos = (w / 2.0).ceil() as i32;
            let y_ext_neg = -((h / 2.0).floor() as i32);
            let y_ext_pos = (h / 2.0).ceil() as i32;

            for (grid_point, local_point) in point_range(-5, 5, -5, 5)
                .cartesian_product(point_range(x_ext_neg, x_ext_pos, y_ext_neg, y_ext_pos))
            {
                let grid_point_pos = grid_point * grid_incr;
                let local_point_pos = local_point + grid_point_pos;
                assert_eq!(grid_point_pos, local_point_pos.snap(&grid_incr));
            }
        }
    }

    #[test]
    fn circle_bresenham_zero_radius() {
        let mut it = Circle::new(5, -4, 0).bresenham_iter();
        assert_eq!(it.next(), Some(Point { x: 5, y: -4 }));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn circle_bresenham_one_radius() {
        let mut it = Circle::new(5, -4, 1).bresenham_iter();
        assert_eq!(it.next(), Some(Point { x: 6, y: -4 }));
        assert_eq!(it.next(), Some(Point { x: 5, y: -3 }));
        assert_eq!(it.next(), Some(Point { x: 4, y: -4 }));
        assert_eq!(it.next(), Some(Point { x: 5, y: -5 }));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn circle_bresenham() {
        for r in 2..=10 {
            let circle_int = Circle::new(0, 0, r);
            let circle_f = Circle::new(0.0, 0.0, r as f32);
            let ring_int = circle_int.bresenham_iter();
            let ring_f = circle_f.bresenham_iter();

            assert!(ring_int.clone().all_unique());

            for (p_int, p_f) in ring_int.zip(ring_f) {
                assert_eq!(p_int.map(|x| x as f32), p_f);

                // check that it's the closest approximation to a point in the circle
                let actual_distance = p_f.distance_euclid();
                let error = (actual_distance - r as f32).abs();
                assert!(error < 1.0);
            }
        }
    }
}
