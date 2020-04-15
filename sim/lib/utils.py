from functools import wraps


def enforce_init_run(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self._is_init:
            return fn(self, *args, **kwargs)
        else:
            raise Exception(('Model is not properly set. '
                             '`set_data` must be called first.'))
    return wrapped
