use crate::{Point, Rect};

use num::{traits::AsPrimitive, Integer, Num, One, Signed, Zero};
use serde::{Deserialize, Serialize};
use std::{fmt, iter::FusedIterator};

/// A generic line segment from the point `(x1, y1)` to `(x2, y2)`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(
    bound(serialize = "T: Clone + Serialize"),
    into = "[T; 4]",
    from = "[T; 4]"
)]
pub struct Line<T> {
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

impl<T> Line<T> {
    /// Creates a new line segment.
    pub fn new(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Creates a new line segment between the given points.
    pub fn from_points(a: Point<T>, b: Point<T>) -> Self {
        (a, b).into()
    }

    /// Converts self into a tuple.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_tuple(self) -> (T, T, T, T) {
        self.into()
    }

    /// Converts self into a tuple of points.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_points(self) -> (Point<T>, Point<T>) {
        (Point::new(self.x1, self.y1), Point::new(self.x2, self.y2))
    }

    /// Converts self into an array.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_array(self) -> [T; 4] {
        self.into()
    }

    /// Converts self into an array of points.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_points_array(self) -> [Point<T>; 2] {
        [Point::new(self.x1, self.y1), Point::new(self.x2, self.y2)]
    }

    /// Returns true if the ending point has greater or equal x-coordinate than the beginning.
    pub fn is_x_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.x1 <= self.x2
    }

    /// Returns true if the ending point has greater or equal y-coordinate than the beginning.
    pub fn is_y_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.y1 <= self.y2
    }

    /// Maps a function to all of the coordinates and returns them as a new line.
    #[must_use]
    pub fn map<F, U>(self, f: F) -> Line<U>
    where
        F: Fn(T) -> U,
    {
        Line {
            x1: f(self.x1),
            y1: f(self.y1),
            x2: f(self.x2),
            y2: f(self.y2),
        }
    }

    /// Maps a function to both of the x coordinates and returns it as a new line.
    #[must_use]
    pub fn map_x<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: f(self.x1),
            y1: self.y1,
            x2: f(self.x2),
            y2: self.y2,
        }
    }

    /// Maps a function to both of the y coordinates and returns it as a new line.
    #[must_use]
    pub fn map_y<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: self.x1,
            y1: f(self.y1),
            x2: self.x2,
            y2: f(self.y2),
        }
    }

    /// Maps a function to the beginning and end point and returns them as a new line.
    #[must_use]
    pub fn map_points<F, U>(self, f: F) -> Line<U>
    where
        F: Fn(Point<T>) -> Point<U>,
    {
        let Line { x1, y1, x2, y2 } = self;
        Line::from_points(f((x1, y1).into()), f((x2, y2).into()))
    }

    /// Maps a function to the beginning point and returns it as a new line.
    #[must_use]
    pub fn map_begin<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: f(self.x1),
            y1: f(self.y1),
            x2: self.x2,
            y2: self.y2,
        }
    }

    /// Maps a function to the ending point and returns it as a new line.
    #[must_use]
    pub fn map_end<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: self.x1,
            y1: self.y1,
            x2: f(self.x2),
            y2: f(self.y2),
        }
    }
}

impl<T: Clone> Line<T> {
    /// Returns the first point `(x1, y1)`.
    pub fn begin(&self) -> Point<T> {
        [self.x1.clone(), self.y1.clone()].into()
    }

    /// Returns the end point `(x2, y2)`.
    pub fn end(&self) -> Point<T> {
        [self.x2.clone(), self.y2.clone()].into()
    }

    /// Returns the midpoint of this line.
    pub fn mid(&self) -> Point<T>
    where
        T: Num,
    {
        let two = || T::one() + T::one();
        let two = || Point::new(two(), two());
        (self.begin() + self.end()) / two()
    }

    /// Checks if both endpoints fulfill the given predicate function
    ///
    /// The predicate is short-circuiting and applied to the beginning point first.
    #[must_use]
    pub fn all<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&Point<T>) -> bool,
    {
        predicate(&self.begin()) && predicate(&self.end())
    }

    /// Checks if either endpoint fulfills the given predicate function
    ///
    /// The predicate is short-circuiting and applied to the beginning point first.
    #[must_use]
    pub fn any<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&Point<T>) -> bool,
    {
        predicate(&self.begin()) || predicate(&self.end())
    }

    /// Returns `end - begin`.
    pub fn vector(&self) -> Point<T>
    where
        T: Num,
    {
        self.end() - self.begin()
    }

    /// Reverses this line to go from `(x2, y2)` to `(x1, y1)`.
    pub fn rev(&self) -> Line<T> {
        let Self { x1, y1, x2, y2 } = self.clone();
        Self::new(x2, y2, x1, y1)
    }

    /// Returns a line with the beginning point having the lesser x-coordinate.
    ///
    /// If the x coordinate of both points is the same, the line is returned unchanged.
    pub fn sort_x(&self) -> Line<T>
    where
        T: PartialOrd,
    {
        if self.is_x_sorted() {
            self.clone()
        } else {
            self.rev()
        }
    }

    /// Returns a line with the beginning point having the lesser y-coordinate.
    ///
    /// If the y coordinate of both points is the same, the line is returned unchanged.
    pub fn sort_y(&self) -> Line<T>
    where
        T: PartialOrd,
    {
        if self.is_y_sorted() {
            self.clone()
        } else {
            self.rev()
        }
    }

    /// Linear interpolation between the two endpoints of this line segment.
    ///
    /// If `t` is 0 then the beginning is returned, if it's 1 then the endpoint is returned
    /// (with some possible floating point inaccuracy).
    ///
    /// Anything between 0 and 1 is a point within this line segment, 0.5 being the middle.
    /// If `t` is less than 0 or bigger than 1 then a point outside the line segment is returned.
    pub fn lerp(&self, t: f32) -> Point<f32>
    where
        T: AsPrimitive<f32> + Num,
    {
        let (x1, y1) = self.begin().into();
        let (dx, dy) = self.vector().into();
        Point {
            x: x1.as_() + dx.as_() * t,
            y: y1.as_() + dy.as_() * t,
        }
    }

    /// Applies the given function to the start and end points of two lines.
    ///
    /// The function is applied to the start point first.
    pub fn op<U, O, F>(&self, other: &Line<U>, mut operator: F) -> Line<O>
    where
        U: Clone,
        F: FnMut(&Point<T>, &Point<U>) -> Point<O>,
    {
        Line::from_points(
            operator(&self.begin(), &other.begin()),
            operator(&self.end(), &other.end()),
        )
    }

    /// Applies the given function to the start and endpoints of many lines.
    ///
    /// The function is applied to the start points first.
    pub fn op_many<'a, I, O, F>(arguments: I, mut operator: F) -> Line<O>
    where
        T: Copy,
        I: IntoIterator<Item = Line<T>>,
        F: FnMut(&[Point<T>]) -> Point<O>,
    {
        let (begin_points, end_points): (Vec<_>, Vec<_>) =
            arguments.into_iter().map(Line::into_points).unzip();
        Line::from_points(operator(&begin_points[..]), operator(&end_points[..]))
    }

    /// Applies the given predicate between the endpoints of two lines and returns if both filled it
    ///
    /// The function is short-circuiting and applied to the start points first.
    pub fn op_all<U, F>(&self, other: &Line<U>, mut predicate: F) -> bool
    where
        U: Clone,
        F: FnMut(&Point<T>, &Point<U>) -> bool,
    {
        predicate(&self.begin(), &other.begin()) && predicate(&self.end(), &other.end())
    }

    /// Applies the given predicate between the endpoints of two lines and returns if either one filled it
    ///
    /// The function is applied to the x-coordinate first.
    pub fn op_any<U, F>(&self, other: &Line<U>, mut predicate: F) -> bool
    where
        U: Clone,
        F: FnMut(&Point<T>, &Point<U>) -> bool,
    {
        predicate(&self.begin(), &other.begin()) || predicate(&self.end(), &other.end())
    }

    /// Returns the length of this line in the euclid space.
    ///
    /// Same as `line.vector().distance_euclid()`.
    pub fn length_euclid(&self) -> f32
    where
        T: AsPrimitive<f32> + Num,
    {
        self.vector().distance_euclid()
    }

    /// Returns the [taxicab/manhattan length](https://en.wikipedia.org/wiki/Taxicab_geometry) of this line.
    ///
    /// Same as [`line.vector().distance_taxi()`](Point::distance_taxi).
    pub fn length_taxi(&self) -> T
    where
        T: Num + PartialOrd,
    {
        self.vector().distance_taxi()
    }

    /// Returns the [Chebyshev/king's move length](https://en.wikipedia.org/wiki/Chebyshev_distance) of this line.
    ///
    /// Same as [`line.vector().distance_king()`](Point::distance_king).
    pub fn length_king(&self) -> T
    where
        T: Num + PartialOrd,
    {
        self.vector().distance_king()
    }

    /// Returns the bounding rectangle of this line.
    ///
    /// Same as `(self.begin(), self.end()).into()`.
    pub fn rect(&self) -> Rect<T>
    where
        T: PartialOrd,
    {
        Rect::from_corners(self.begin(), self.end())
    }

    /// Snaps the endpoints of this line to the nearest multiple of the given increment.
    ///
    /// Halfway cases are rounded down towards negative infinity.
    ///
    /// The increment's sign doesn't affect the result.
    ///
    /// # Panics
    /// Panics if either of the increment's values are 0.
    pub fn snap(&self, snap_increment: &Point<T>) -> Self
    where
        T: Integer,
    {
        self.clone().map_points(|p| p.snap(snap_increment))
    }

    /// Returns an iterator over the points in this line using the Bresenham algorithm.
    ///
    /// The returned points will be on a (1,1)-grid and the returned order will be arbitrary.
    pub fn bresenham_iter(&self) -> BresenhamIter<T>
    where
        T: Signed + PartialOrd,
    {
        let Line { x1, y1, x2, y2 } = {
            let Line { x1, y1, x2, y2 } = self.clone();
            let dx = x2 - x1;
            let dy = y2 - y1;
            // flip if needed
            if dx + dy > T::zero() {
                self.clone()
            } else {
                self.rev()
            }
        };
        let dx = x2.clone() - x1.clone();
        let dy = y2.clone() - y1.clone();

        let u_is_x = dx >= dy;
        let (u, v, u_end, du, dv) = if u_is_x {
            (x1, y1, x2, dx, dy)
        } else {
            (y1, x1, y2, dy, dx)
        };
        let v_increment = if dv >= T::zero() { T::one() } else { -T::one() };
        let du = du.abs();
        let dv = dv.abs();

        BresenhamIter {
            u_is_x,
            v_increment,
            u,
            v,
            u_end,
            du,
            dv,
            err: T::zero(),
        }
    }
}

/// An iterator returning a line's points according to the Bresenham algorithm.
///
/// Created with [`Line::bresenham`].
#[derive(Debug, Clone, Copy)]
pub struct BresenhamIter<T> {
    u_is_x: bool,
    u: T,
    v: T,
    u_end: T,
    v_increment: T,
    du: T,
    dv: T,
    err: T,
}

impl<T> Iterator for BresenhamIter<T>
where
    T: Clone + PartialOrd + One + Zero + Signed,
{
    type Item = Point<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let two = || T::one() + T::one();

        let to_return = if self.u_is_x {
            (self.u.clone(), self.v.clone())
        } else {
            // flip values if needed
            (self.v.clone(), self.u.clone())
        }
        .into();

        if self.u < self.u_end {
            // always move in u-direction
            self.u = self.u.clone() + T::one();

            self.err = self.err.clone() + self.dv.clone();

            if self.err.clone() * two() >= self.du {
                self.err = self.err.clone() - self.du.clone();
                // now move one in v-direction
                self.v = self.v.clone() + self.v_increment.clone();
            }

            Some(to_return)
        } else if self.u == self.u_end {
            // the last point
            self.u = self.u.clone() + T::one();
            Some(to_return)
        } else {
            // done
            None
        }
    }
}

impl<T> FusedIterator for BresenhamIter<T> where T: Clone + PartialOrd + One + Zero + Signed {}

impl<T: fmt::Display> fmt::Display for Line<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}, {}", self.x1, self.y1, self.x2, self.y2)
    }
}

impl<T> From<[T; 4]> for Line<T> {
    fn from([x1, y1, x2, y2]: [T; 4]) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<[[T; 2]; 2]> for Line<T> {
    fn from([[x1, y1], [x2, y2]]: [[T; 2]; 2]) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<[Point<T>; 2]> for Line<T> {
    fn from([begin, end]: [Point<T>; 2]) -> Self {
        let [x1, y1]: [T; 2] = begin.into();
        let [x2, y2]: [T; 2] = end.into();
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<(T, T, T, T)> for Line<T> {
    fn from((x1, y1, x2, y2): (T, T, T, T)) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<((T, T), (T, T))> for Line<T> {
    fn from(((x1, y1), (x2, y2)): ((T, T), (T, T))) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<(Point<T>, Point<T>)> for Line<T> {
    fn from((begin, end): (Point<T>, Point<T>)) -> Self {
        [begin, end].into()
    }
}

impl<T> From<Line<T>> for [T; 4] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [x1, y1, x2, y2]
    }
}

impl<T> From<Line<T>> for [[T; 2]; 2] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [[x1, y1], [x2, y2]]
    }
}

impl<T> From<Line<T>> for [Point<T>; 2] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [Point::new(x1, y1), Point::new(x2, y2)]
    }
}

impl<T> From<Line<T>> for (T, T, T, T) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        (x1, y1, x2, y2)
    }
}

impl<T> From<Line<T>> for ((T, T), (T, T)) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        ((x1, y1), (x2, y2))
    }
}

impl<T> From<Line<T>> for (Point<T>, Point<T>) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        (Point::new(x1, y1), Point::new(x2, y2))
    }
}
