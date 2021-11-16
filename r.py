from tqdm import tqdm
from time import sleep
epoch = 0
with tqdm(total=100,desc=f'Epoch {epoch + 1}/{100}',postfix=dict,mininterval=0.3, bar_format = '{l_bar}{bar}| [{n_fmt}/{total_fmt}{postfix}]') as pbar:
    for i in range(100):
        pbar.set_postfix(**{'loss'  : epoch / (99 + 1), 
                        'lr'    : epoch})
        pbar.update(1)
        if i == 50:
            pbar.set_description(f'Epoch {epoch + 2}/{100}')
        sleep(0.1)