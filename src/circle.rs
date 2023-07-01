use crate::Point;

use num::{Integer, Num, Signed};
use serde::{Deserialize, Serialize};
use std::{fmt, iter::FusedIterator};

/// A generic circle centered around the point `(x, y)` with the radius `r`.
///
/// If the radius is negative, then many of the methods might panic or return nonsensical answers.
/// Not all methods might check for this.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Circle<T> {
    pub x: T,
    pub y: T,
    pub r: T,
}

impl<T> Circle<T> {
    /// Creates a new circle
    pub const fn new(x: T, y: T, r: T) -> Self {
        Self { x, y, r }
    }

    /// Maps the given function to this circle's center and returns a new circle
    pub fn map_xy<F>(self, mut f: F) -> Self
    where
        F: FnMut(Point<T>) -> Point<T>,
    {
        let Point { x, y } = f([self.x, self.y].into());
        Self { x, y, r: self.r }
    }

    /// Maps the given function to this circle's radius and returns a new circle
    pub fn map_r<F>(self, mut f: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        Self {
            x: self.x,
            y: self.y,
            r: f(self.r),
        }
    }

    /// Maps the given function to this circle's position and radius and returns a new circle
    ///
    /// The function is applied first to the x-coordinate, then to the y-coordinate, and last
    /// to the radius.
    pub fn map_xyr<F, U>(self, mut f: F) -> Circle<U>
    where
        F: FnMut(T) -> U,
    {
        Circle {
            x: f(self.x),
            y: f(self.y),
            r: f(self.r),
        }
    }
}
impl<T: Clone> Circle<T> {
    /// Returns the center point of this circle as a `Point`
    pub fn center(&self) -> Point<T> {
        Point {
            x: self.x.clone(),
            y: self.y.clone(),
        }
    }

    /// Returns true if the given point is inside this circle
    ///
    /// Is true if the euclidean distance of the given point from the circle's center
    /// is less than or equal to the circle's radius, plus 1.
    pub fn is_inside(&self, point: &Point<T>) -> bool
    where
        T: Num + PartialOrd,
    {
        let v = point - self.center();
        v.dot(&v) <= self.r.clone() * self.r.clone() + T::one()
    }

    /// Returns an iterator over the points of the circle's ring using integer arithmetic
    ///
    /// The iterator returns the point `(radius, 0)` first, and then proceeds to return the
    /// rest of the points in the counter-clockwise order, if y-coordinate is up.
    ///
    /// If the radius is 0, then the resulting iterator will return just the circle's center.
    /// Radius 1 circle is a 3x3 "diamond".
    ///
    /// # Panics
    /// Panics if the radius is negative.
    pub fn bresenham_iter(&self) -> BresenhamCircleIter<T>
    where
        T: Num + PartialOrd + Signed + Integer,
    {
        BresenhamCircleIter {
            center: self.center(),
            octant_points: self.bresenham_arc(false).into_boxed_slice(),
            current_octant: 0,
            current_index: 0,
        }
    }

    /// Returns an iterator over the points of the filled circle using integer arithmetic
    ///
    /// The iterator returns the points in the order of increasing coordinates
    ///
    /// If the radius is 0, then the resulting iterator will return just the circle's center.
    /// Radius 1 circle is a 3x3 "plus sign".
    ///
    /// # Panics
    /// Panics if the radius is negative.
    pub fn bresenham_filled_iter(&self) -> BresenhamFilledCircleIter<T>
    where
        T: Num + PartialOrd + Signed + Integer,
    {
        let (current_pos, segment_points) = if self.r.is_zero() {
            (self.center(), Vec::new())
        } else {
            let pts = self.bresenham_arc(true);
            (-Point::from(pts.last().unwrap().clone()), pts)
        };

        BresenhamFilledCircleIter {
            center: self.center(),
            current_line: -(segment_points.len() as isize - 1),
            current_pos,
            quadrant_points: segment_points.into_boxed_slice(),
        }
    }

    /// The actual bresenham algorithm is performed here beforehand
    fn bresenham_arc(&self, fill_mode: bool) -> Vec<Point<T>>
    where
        T: Num + PartialOrd + Signed + Integer,
    {
        assert!(!self.r.is_negative(), "a circle's radius was negative");

        let two = || T::one() + T::one();
        let three = || two() + T::one();
        let five = || two() + three();

        let mut octant_points = Vec::new(); // the circle arc for the octant
        let mut x = self.r.clone();
        let mut y = T::zero();

        // the arc from 45° to 90° with all-unique y-coordinates, required for fill
        let mut fill_octant = Vec::new();

        let mut error_delta = three() - two() * x.clone();

        // fill the octant circle arc until you reach the line y = x
        while y <= x {
            octant_points.push((x.clone(), y.clone()).into());
            if error_delta.is_positive() {
                // time to move x one step back
                if fill_mode {
                    fill_octant.push((y.clone(), x.clone()).into());
                }
                x = x - T::one();
                error_delta =
                    error_delta + two() * (five() - two() * x.clone() + two() * y.clone());
            } else {
                error_delta = error_delta + two() * (three() + two() * y.clone());
            }
            y = y + T::one();
        }

        if fill_mode {
            // combine the 2 octants for a quadrant
            fill_octant.reverse();
            octant_points.extend_from_slice(&fill_octant);
        }

        octant_points
    }
}

/// An iterator returning the circle's points according to the Bresenham's circle algorithm
///
/// Created by [`Circle::bresenham_iter`].
#[derive(Debug, Clone)]
pub struct BresenhamCircleIter<T> {
    center: Point<T>,
    octant_points: Box<[Point<T>]>,
    current_octant: u8,
    current_index: usize,
}

impl<T> Iterator for BresenhamCircleIter<T>
where
    T: Clone + Num + PartialOrd + Signed + Integer,
{
    type Item = Point<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_octant >= 8 {
            return None;
        }

        if let [Point { x: r, .. }] = self.octant_points.as_ref() {
            // radius is either 1 or 0
            if r.is_zero() {
                // radius is 0
                self.current_octant = 8;
                return Some(self.center.clone());
            } else {
                // radius is 1
                let (x, y) = self.center.clone().into_tuple();
                let out = Point::from(match self.current_octant {
                    0 => (x + r.clone(), y),
                    2 => (x, y + r.clone()),
                    4 => (x - r.clone(), y),
                    6 => (x, y - r.clone()),
                    _ => unreachable!(),
                });
                self.current_octant += 2;

                return Some(out);
            }
        }

        let is_flipped = self.current_octant & 1 == 1;

        let idx = if !is_flipped {
            self.current_index
        } else {
            // a flipped octant
            // the arc is read in the reverse order
            self.octant_points.len() - self.current_index - 1
        };

        let (prim, sec) = self.octant_points[idx].clone().into_tuple();

        // assign the primary and secondary axises
        let (dx, dy) = match self.current_octant {
            7 | 0 | 3 | 4 => (prim, sec),
            _ => (sec, prim),
        };

        // get the correct signs
        let x_sign = match self.current_octant {
            6 | 7 | 0 | 1 => T::one(),
            _ => -T::one(),
        };
        let y_sign = match self.current_octant {
            0 | 1 | 2 | 3 => T::one(),
            _ => -T::one(),
        };

        self.current_index += 1;
        if self.current_index >= self.octant_points.len() - is_flipped as usize {
            // one octant finished
            self.current_index = 0;
            self.current_octant += 1;
        }

        let (cx, cy) = self.center.clone().into_tuple();
        Some((cx + x_sign * dx, cy + y_sign * dy).into())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left =
            (8 - self.current_octant) as usize * self.octant_points.len() - self.current_index;
        (left, Some(left))
    }
}

impl<T> FusedIterator for BresenhamCircleIter<T> where T: Clone + Num + PartialOrd + Signed + Integer
{}
impl<T> ExactSizeIterator for BresenhamCircleIter<T> where
    T: Clone + Num + PartialOrd + Signed + Integer
{
}

/// An iterator returning the filled circle's points according to the Bresenham's circle algorithm
///
/// Created by [`Circle::bresenham_filled_iter`].
#[derive(Debug, Clone)]
pub struct BresenhamFilledCircleIter<T> {
    center: Point<T>,
    quadrant_points: Box<[Point<T>]>,
    current_line: isize,
    current_pos: Point<T>,
}

impl<T> Iterator for BresenhamFilledCircleIter<T>
where
    T: Clone + Num + PartialOrd + Signed + Integer,
{
    type Item = Point<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.quadrant_points.is_empty() && self.current_line != 0 {
            self.current_line = 0;
            return Some(self.center.clone());
        }

        if self.current_line >= self.quadrant_points.len() as isize {
            return None;
        }

        let get_quadrant_point = |line: isize| self.quadrant_points[line.abs() as usize].clone();

        let out = self.center.clone() + self.current_pos.clone();

        let (max_x, _) = get_quadrant_point(self.current_line).into_tuple();

        self.current_pos.x = self.current_pos.x.clone() + T::one();
        if self.current_pos.x > max_x {
            // one line finished
            self.current_line += 1;
            // check if it was last line
            if self.current_line < self.quadrant_points.len() as isize {
                self.current_pos = get_quadrant_point(self.current_line)
                    .map_x(|x| -x)
                    .map_y(|y| if self.current_line < 0 { -y } else { y });
            }
        }

        Some(out)
    }
}

impl<T> FusedIterator for BresenhamFilledCircleIter<T> where
    T: Clone + Num + PartialOrd + Signed + Integer
{
}

impl<T: fmt::Display> fmt::Display for Circle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x: {}, y: {}, r: {}", self.x, self.y, self.r)
    }
}
