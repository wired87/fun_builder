try:
    import jax
    import jax.numpy as jnp

    LIBS={
        "jax": jax,
        "vmap": jax.vmap,
        "jnp": jnp,
        "jit": jax.jit,
    }
except Exception as e:
    LIBS= {}
    
def create_runnable(eq_code:str, libs:dict):
    """
    Create runnable based on given code str
    """
    try:
        namespace = {}

        # Wir fügen die LIBS direkt in den globalen Scope des exec ein
        exec(eq_code, libs or LIBS, namespace)

        # Filtere alle Funktionen heraus
        callables = {
            k: v for k, v in namespace.items()
            if callable(v) and not k.startswith("__")
        }

        if not callables:
            raise ValueError("No fun for you this time...")

        func_name = list(callables.keys())[-1]
        func = callables[func_name]
        # run it with func(**pckg)
        return func
    except Exception as e:
        print(f"Err create_runnable: {e}")
    return None

