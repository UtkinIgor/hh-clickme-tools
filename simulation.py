import numpy as np
from utils import softmax, get_best_banner, get_stats, minmax_scale

# init stats
Name = ['adam','bob','cortny','deni','eva','fred','greg','hugo','iris','jeck','ken','lisa','mike','nik','opra','pit','rick','sam','tim']
banner = {key:[0,0,0,0] for key in Name}

# init params
T = 0.05
SCALE = True
command = 0

while command != 'esc':

    if command == '+':
        T += 0.01
    elif command == '-':
        T -= 0.01
    elif command == 'scale_true':
        SCALE = True
    elif command == 'scale_false':
        SCALE = False

    # Что показываем
    for i, key in enumerate(get_best_banner(banner, top=4, t=T, scale=SCALE)):
        banner[key][0] += 1
        print(i, key)

    # Update stats
    ctr = get_stats(banner)
    if SCALE:
        prob = softmax(minmax_scale(ctr), t=T)
    else:
        prob = softmax(ctr, t=T)

    for i, key in enumerate(banner):
        banner[key][2] = round(ctr[i], 6)
        banner[key][3] = round(prob[i], 6)

    command = input('Your choice: ')

    # вывод статистики
    if command == 'stat':
        print('\n','='*50, f'Temperature is = {T}', sep='\n')
        print('Scale is =', SCALE, end='\n'*2)
        print ("{:<6} {:<6} {:<6} {:<10} {:<10}".format('Key','view','click', 'ctr', 'prob'))
        for k, v in banner.items():
            _view, _click, _ctr, _prob = v
            print("{:<6} {:<6} {:<6} {:<10} {:<10}".format(k, _view, _click, _ctr, _prob))
        print('='*50)

    # Счетчик кликов
    if command in banner.keys():
        banner[command][1] += 1

    print('', end = '\n')
