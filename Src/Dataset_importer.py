import numpy as np


# заменяет все запятые в строке точками
def __convert_colons(string):
    return string.replace(',', '.')


def import_reds():
    dataset_red_inputs = list()
    dataset_red_outputs = list()
    with open('Dataset_red.txt', encoding='UTF') as f:
        for line in f:
            data_line = line.split('\t')
            # перевод текстовых выражених числовых переменных в собственно числа
            for number, item in enumerate(data_line[:15]):
                data_line[number] = float(__convert_colons(item))
            # отрубание перехода на следующую строку в последнем элементе
            data_line[16] = data_line[16][:-1]
            # добавления вектора входов
            dataset_red_inputs.append(data_line[:15])
            # добавление вектора выходов
            dataset_red_outputs.append(data_line[15:])
    return np.array(dataset_red_inputs), np.array(dataset_red_outputs)


def import_whites():
    dataset_white_inputs = list()
    dataset_white_outputs = list()
    # методы тербуют, чтобы метки были заменены числами
    type_map = dict()
    zone_map = dict()
    with open('Dataset_white.txt', encoding='UTF') as f:
        for line in f:
            data_line = line.split('\t')
            # перевод текстовых выражених числовых переменных в собственно числа
            for number, item in enumerate(data_line[:15]):
                data_line[number] = float(__convert_colons(item))
            # отрубание перехода на следующую строку в последнем элементе
            data_line[16] = data_line[16][:-1]
            # добавления вектора входов
            dataset_white_inputs.append(data_line[:15])
            # добавление вектора выходов
            dataset_white_outputs.append(data_line[15:])
    return np.array(dataset_white_inputs), np.array(dataset_white_outputs),


if __name__ == '__main__':
    print(import_reds()[1])
    print(import_whites()[1])
