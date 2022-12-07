import logging
from copy import deepcopy
from typing import Tuple, Type, TypeVar, Union

import numpy as np

from .axis import ArrayAxis
from .config.base_config import Config
from .core import MathOperation, RelationProtocol
from .exc import BadInputError, NotEqualError, TypeFuncError
from .help_types import ArrayLike, Number, RealNumber

R = TypeVar("R", bound="Relation")
'''Description first `Relation`.'''

R2 = TypeVar("R2", bound="Relation")
'''Description second `Relation`.'''


class Relation(RelationProtocol):
    '''A representation of dependency y from x (y = f(x)).

    The class describe the dependency between x, y. x is `ArrayAxis` instance showing
    the start of sequence, the end of sequence and the sample space between elements.
    They consist of real or complex numbers. The array length calculated from
    `ArrayAxis` must be equal length of y sequence.

    For the instance of `Relation` class, define the basic mathematical operations:
    addition (+), subtraction(-), multiplication(\\*), division(/),
    exponentiation (\\*\\*) and their unary representation (+=, -=, \\*=, /=).
    The result of the operation is a new instance of the `Relation` class.

    Determined correlation and convolution between two instances
    (methods: correlate and convolve).

    How those operations will be calculated determined by the methods described
    in the Config class. Methods can be overridden if necessary
    (sweep-design.config).

    WARNING!!! When inheriting the `Relation` class, it is important to write correctly
    constructor. It must match the constructor of the `Relation` class.
    Because some methods return a type(self)(...). For example,
    addition method (def __add__(self: R, other: Union['Relation', Num]) -> R).
    Or predefine these methods in the inherited class.

    Raises:
        BadInputError: Raise this exception if we don't have enough data.
        NotEqualError: Raise this exception if we try create instance use
            different length of sequence numbers for x and y.
        TypeFuncError: Raise an exception, when execute some function with
            unexpected type of value.

    Returns:
        _type_: Type of Relation.
    '''

    def __init__(
        self,
        x: Union[RelationProtocol, ArrayAxis, ArrayLike],
        y: ArrayLike = None,
    ) -> None:
        '''Initialization of instance of `Relation`.

        Args:
            x (Union[RelationProtocol, ArrayLike, ArrayAxis]):
                The `Relation` class, or a class derived from the `Relation`
                class, or instance of `ArrayAxis` or an `ArrayLike` object
                containing numbers(real or complex). if x is `ArrayLike` then
                it will be converted to `ArrayAxis` instance use method
                *get_array_axis_from_array_method* from Config class

            y (ArrayLike, optional):
                None or array_like object containing real or complex numbers.
                If it is not None then it will be converted to np.ndarray.
                Defaults to None.

        Raises:
            BadInputError:Raise this exception if we don't have enough data.
            NotEqualError: Raise this exception if we try create instance use
        '''

        self._get_array_axis_from_array_method = Config.get_array_axis_from_array_method
        self._math_operation = Config.math_operation
        self._interpolate_extrapolate_method = Config.interpolate_extrapolate_method
        self._integrate_one_method = Config.integrate_one_method
        self._integrate_method = Config.integrate_method
        self._differentiate_method = Config.differentiate_method

        if isinstance(x, RelationProtocol):
            self._x = x.x.copy()
            self._y = x.y.copy()
            if y is not None:
                logging.warning(f'x is instance of {type(x)}, "y" was ignored')
            return None

        if y is None:
            raise BadInputError("y is absent. Not enough data!")

        y = np.array(y)

        if not isinstance(x, ArrayAxis):
            x = self._get_array_axis_from_array_method(x)

        if x.size != y.size:
            raise NotEqualError(x.size, y.size)

        self._x, self._y = x, y

    @property
    def x(self) -> ArrayAxis:
        '''ArrayAxis of relation.

        Returns:
            ArrayAxis: array axis of relation.
        '''
        return self._x

    @property
    def y(self) -> np.ndarray:
        '''Result of relation of y(x)

        Returns:
            np.ndarray: array of numbers represent relation of y(x)
        '''
        return self._y

    @property
    def start(self) -> RealNumber:
        '''Start of array axis x.

        Returns:
            RealNumber: start number of array axis x.
        '''
        return self._x.start

    @start.setter
    def start(self, value: RealNumber) -> None:
        '''Setter for start.

        Args:
            value (RealNumber): set start for array axis x.
        '''
        self._x.start = value

    @property
    def end(self) -> RealNumber:
        '''End of array axis x.

        Returns:
            RealNumber: end number of array axis x.
        '''
        return self._x.end

    @end.setter
    def end(self, value: RealNumber):
        '''Setter for start.

        Args:
            value (RealNumber): set end for array axis x.
        '''
        self._x.end = value

    @property
    def sample(self) -> RealNumber:
        '''Sample for array axis x.

        Returns:
            RealNumber: sample of array axis x.
        '''
        return self._x.sample

    @sample.setter
    def sample(self, value: RealNumber) -> None:
        '''Setter for sample.

        Args:
            value (RealNumber): set sample for array axis x.
        '''
        self._x.sample = value

    @property
    def array(self) -> np.ndarray:
        '''Get array representation of array axis x.

        Returns:
            np.ndarray: array of numpy.
        '''
        return self._x.array

    @property
    def actual_sample(self) -> Number:
        '''Get actual sample or array axis x.

        Returns:
            Number: number of actual sample array x.
        '''
        return self._x.actual_sample

    @property
    def size(self) -> int:
        '''size of array axis x.

        Returns:
            int: integer number of array size x.
        '''
        return self._x.size

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Return the data of the object.

        Raises:
            NotEqualError: After manipulating on x ArrayAxis, the size of the
                extracted arrays is checked. If they are different then raise
                that error.

        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of two number sequence
        '''

        if self._x.size != self._y.size:
            raise NotEqualError(self._x.size, self._y.size)

        return self.array.copy(), self.y.copy()

    def max(self) -> Number:
        '''Get maximum of Relation.

        Returns:
            Number: maximum of y array.
        '''
        return self._y.max()

    def min(self) -> Number:
        '''Get minimum of Relation.

        Returns:
            Number: minimum of y array.
        '''
        return self._y.min()

    def get_norm(self) -> RealNumber:
        '''Get signal rate.

        Calculated in terms of signal energy.

        Returns:
            Number: signal rate
        '''

        square_self = self**2
        return self._integrate_one_method(square_self) / (self.sample)

    def select_data(self: R, start: Number = None,
                    end: Number = None) -> R:
        '''Select data using x-axis

        Args:
            self (R): instance of Relation
            start (Number, optional): new start of relation x. Defaults to None.
            end (Number, optional): new end of relation x. Defaults to None.

        Returns:
            R: new instance of Relation.
        '''

        if start is None:
            start = self.start

        if end is None:
            end = self.end

        array = self.array

        is_selected = np.logical_and(
            np.greater_equal(array, start), np.less_equal(array, end)
        )
        selected_x = array[is_selected]

        new_x_array = ArrayAxis(
            start=selected_x[0], end=selected_x[-1], sample=self.sample)

        return type(self)(new_x_array, self.y[is_selected])

    def exp(self: R) -> R:
        '''Get exponent of Relation.

        Args:
            self (R): instance of Relation

        Returns:
            R: Relation where new y is exponent of old y.
        '''
        return type(self)(self.x.copy(), np.exp(self.y))

    def diff(self: R) -> R:
        '''Differentiation of 'Relation'.

        Args:
            self (R): instance of Relation

        Returns:
            R: result of differentiation.
        '''
        result = self._differentiate_method(self)
        return type(self)(*result)

    def integrate(self: R) -> R:
        '''Integration of `Relation`.

        Args:
            self (R): instance of Relation

        Returns:
            R: result of cumulative integration.
        '''
        result = self._integrate_method(self)
        return type(self)(*result)

    def interpolate_extrapolate(
            self: R, new_x: Union[R, ArrayAxis, ArrayLike]) -> R:
        '''Interpolates and extrapolates an existing relation using new array
        x of the represented ArrayAxis instance.

        Args:
            self (R): instance of Relation
            new_x (ArrayAxis): new x array axis

        Returns:
            R: new instance of Relation
        '''
        if isinstance(new_x, Relation):
            new_x = new_x.x.copy()
        elif isinstance(new_x, ArrayAxis):
            new_x = new_x.copy()
        else:
            new_x = self._get_array_axis_from_array_method(new_x, False)

        new_y = self._interpolate_extrapolate_method(
            self.x.array.copy(), self.y.copy())(new_x)
        return type(self)(new_x, new_y)

    def shift(self: R, x_shift: RealNumber = 0) -> R:
        '''Shifting of relation on the x-axis.

        Args:
            self (R): instance of Relation
            x_shift (Number, optional): Number of displacement on the x-axis.
            Defaults to 0.

        Returns:
            R: new instance of Relation
        '''
        new_x = self.x.copy()
        new_x.start = new_x.start + x_shift
        new_x.end = new_x.end + x_shift
        return type(self)(new_x, self.y)

    @staticmethod
    def equalize(r1: R, r2: R2) -> Tuple[R, R2]:
        '''Bringing two Relation objects with different x-axes to one common one.

        When converting, interpolation and extrapolation are used.

        Args:
            r1 (R): first instance of Relation
            r2 (R2): second instance of Relation

        Returns:
            Tuple[R, R2]: tuple of new Relation instances with common axis.
        '''
        if r1.x == r2.x:
            return deepcopy(r1), deepcopy(r2)

        new_x = Config.get_common_x(r1.x, r2.x)
        r1 = r1.interpolate_extrapolate(new_x)
        r2 = r2.interpolate_extrapolate(new_x)

        return r1, r2

    @classmethod
    def correlate(cls: Type[R], r1: "Relation", r2: "Relation") -> R:
        '''Correlation of two Relations.

        Args:
            cls (Type[R]): class of Relation
            r1 (Relation): first Relation.
            r2 (Relation): second Relation.

        Raises:
            TypeFuncError: raise exception
            if we try correlate with unexpected types.

        Returns:
            R: new instance of Relation
        '''

        if isinstance(r1, Relation) and isinstance(r2, Relation):
            result = Config.correlate_method(cls, r1, r2)
            return cls(*result)
        else:
            raise TypeFuncError("Correlation", type(r1), type(r2))

    @classmethod
    def convolve(cls: Type[R], r1: "Relation", r2: "Relation") -> R:
        '''Convolution of two Relations.

        Args:
            cls (Type[R]): class of Relation
            r1 (Relation): first Relation.
            r2 (Relation): second Relation.

        Raises:
            TypeFuncError: raise exception
            if we try correlate with unexpected types.

        Returns:
            R: new instance of Relation
        '''
        if isinstance(r1, Relation) and isinstance(r2, Relation):
            result = Config.convolve_method(cls, r1, r2)
            return cls(*result)
        else:
            raise TypeFuncError("Convolution", type(r1), type(r2))

    @staticmethod
    def _operation(
        a: "Relation", b: Union["Relation", Number], name_operation: MathOperation
    ) -> Tuple[ArrayAxis, np.ndarray]:
        logging.debug(f"Type of a: {type(a)}")
        logging.debug(f"Type of b: {type(b)}")

        if isinstance(b, RelationProtocol):
            r1, r2 = Relation.equalize(a, b)
            return r1.x.copy(), a._math_operation(
                r1.y.copy(), r2.y.copy(), name_operation)
        else:
            return a.x.copy(), a._math_operation(
                a.y.copy(), b, name_operation)

    def __add__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.ADD))

    def __radd__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(*self._operation(self, other,
                                           MathOperation.RADD))

    def __sub__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.SUB))

    def __rsub__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(*self._operation(self, other,
                                           MathOperation.RSUB))

    def __mul__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.MUL))

    def __rmul__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(*self._operation(self, other,
                                           MathOperation.RMUL))

    def __truediv__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.TRUEDIV)
        )

    def __rtruediv__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.RTRUEDIV)
        )

    def __pow__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(
            *self._operation(self, other, MathOperation.POW))

    def __rpow__(self: R, other: Union["Relation", Number]) -> R:
        return type(self)(*self._operation(self, other,
                                           MathOperation.RPOW))

    def __iadd__(self: R, other: Union["Relation", Number]) -> R:
        return self.__add__(other)

    def __isub__(self: R, other: Union["Relation", Number]) -> R:
        return self.__sub__(other)

    def __imul__(self: R, other: Union["Relation", Number]) -> R:
        return self.__mul__(other)

    def __idiv__(self: R, other: Union["Relation", Number]) -> R:
        return self.__truediv__(other)

    def __ipow__(self: R, other: Union["Relation", Number]) -> R:
        return self.__pow__(other)

    def __len__(self) -> int:
        return self._x.size

    def __getitem__(
            self: R, select_data: Union[Number, slice]) -> Union[Tuple[Number, Number], R]:
        '''Select data from Relation

        if item is Number then function return tuple of two numbers.
        The first number is number near to select data.
        Second number is number represent of relation to selected data.

        if select data is slice then function return Relation that
        equal Relation if we call select_data function of instance.

        Args:
            self (R): instance of Relation
            item (Union[float, slice]): selected data is number or slice

        Returns:
            Union[Tuple[Num, Num], R]: two number or instance of relation.
        '''

        if isinstance(select_data, (float, int, complex)):
            array = self.array
            idx = (np.abs(array - select_data)).argmin()
            return array[idx], self.y[idx]

        if isinstance(select_data, slice):
            return self.select_data(select_data.start, select_data.stop)

    def __str__(self) -> str:
        return f"y: {self.y}\nx: {str(self.x)}"
