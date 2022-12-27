use crate::Point;

use num::{Num, Signed};
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
    pub fn new(x: T, y: T, r: T) -> Self {
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
    /// is less than or equal to the circle's radius.
    pub fn is_inside(&self, point: &Point<T>) -> bool
    where
        T: Num + PartialOrd,
    {
        let v = point - self.center();
        v.dot(&v) <= self.r.clone() * self.r.clone()
    }

    /// Returns an iterator over the points of the circle's ring using integer arithmetic
    ///
    /// The iterator returns the point `(radius, 0)` first, and then proceeds to return the
    /// rest of the points in the counter-clockwise order, if y-coordinate is up.
    ///
    /// If the radius is 0, then the resulting iterator will return just the circle's center.
    /// Radius 1 circle is a 3x3 "plus sign".
    ///
    /// # Panics
    /// Panics if the radius is negative.
    pub fn bresenham_iter(&self) -> BresenhamCircleIter<T>
    where
        T: Num + PartialOrd + Signed,
    {
        assert!(!self.r.is_negative(), "a circle's radius was negative");

        let two = || T::one() + T::one();
        let three = || two() + T::one();
        let five = || two() + three();

        let mut octant_points = Vec::new(); // the circle arc for the octant
        let mut x = self.r.clone();
        let mut y = T::zero();

        let mut error_delta = three() - two() * x.clone();

        // fill the octant circle arc until you reach the line y = x
        while y <= x {
            octant_points.push((x.clone(), y.clone()));
            if error_delta.is_positive() {
                // time to move x one step back
                x = x - T::one();
                error_delta =
                    error_delta + two() * (five() - two() * x.clone() + two() * y.clone());
            } else {
                error_delta = error_delta + two() * (three() + two() * y.clone());
            }
            y = y + T::one();
        }

        BresenhamCircleIter {
            center: self.center(),
            octant_points: octant_points.into_boxed_slice(),
            current_octant: 0,
            current_index: 0,
        }
    }
}

/// An iterator returning the circle's points according to the Bresenham's circle algorithm
///
/// Created by [`Circle::bresenham_iter`].
#[derive(Debug, Clone)]
pub struct BresenhamCircleIter<T> {
    center: Point<T>,
    octant_points: Box<[(T, T)]>,
    current_octant: u8,
    current_index: usize,
}

impl<T> Iterator for BresenhamCircleIter<T>
where
    T: Clone + Num + PartialOrd + Signed,
{
    type Item = Point<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_octant >= 8 {
            return None;
        }

        if let p @ [(r, _)] = self.octant_points.as_ref() {
            // radius is either 1 or 0
            if p == [(T::zero(), T::zero())] {
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

        let (prim, sec) = self.octant_points[idx].clone();

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
        if self.current_index >= self.octant_points.len() - 1 {
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

impl<T> FusedIterator for BresenhamCircleIter<T> where T: Clone + Num + PartialOrd + Signed {}
impl<T> ExactSizeIterator for BresenhamCircleIter<T> where T: Clone + Num + PartialOrd + Signed {}

impl<T: fmt::Display> fmt::Display for Circle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x: {}, y: {}, r: {}", self.x, self.y, self.r)
    }
}
