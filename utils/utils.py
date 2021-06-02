def iter_to_str(iterable):
    """
    Writes iterable object into string separating items with spaces.

    Parameters
    ----------
    iterable : iterable

    Returns
    -------
    str
    """
    return ' '.join(map(str, iterable))


def documentation_inheritance(base):
    """
    Decorator for copying documentation.

    Parameters
    ----------
    base : Any
        Original object.

    Returns
    -------
    Any
        Decorated object.
    """

    def wrapper(inheritor):
        inheritor.__doc__ = base.__doc__
        return inheritor

    return wrapper
