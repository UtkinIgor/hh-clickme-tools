import numpy as np

def minmax_scale(data: list) -> np.ndarray:
    diff = np.max(data) - np.min(data)
    if diff:
        return (data - np.min(data))/diff
    else:
        return [0]*len(data)

def get_stats(data: dict) -> list:
    return [click/view if view > 0 else 0 for view, click, ctr, prob in data.values()]

def softmax(data: list, t=1) -> np.ndarray:
    trans = np.array(data, dtype=float)/t
    e = np.exp(trans - np.max(trans))
    return e/np.sum(e)

def get_best_banner(data: dict, top=3, t=1, scale=False) -> list:
    """
       Выбор заданного кол-ва элементов словаря согласно их вероятностям
       Params:
            top - кол-во элеметов которые вернет функция.
            t - температурный коэф. который будет передан функции расчета вероятности.
            scale - если True то веса элементов будут приведены к интервалу [0,1] 
                    перед расчетом вероятности.
    """

    # если кандидатов меньше чем рекламных мест то без расчета вернем всех кандидатов
    if len(data)<top:
        return list(data.keys())

    # масштабируем веса если требуется
    if scale:
        weigth = minmax_scale(get_stats(data))
    else:
        weigth = get_stats(data)

    res = np.random.choice(a=list(data.keys()), size=top, replace=False, p=softmax(weigth, t=t))
    return list(res)
