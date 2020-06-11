from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC


class Models(object):
    def __init__(self):
        # словарь в котором дожны будут храниться указатели на все используемые модели
        # должны присутствовать white_svm_type, white_svm_zone,
        # white_nbc_type, white_nbc_zone,
        # white_knn_type, white_knn_zone,
        # red_svm_type, red_svm_zone,
        # red_nbc_type, red_nbc_zone,
        # red_knn_type, red_knn_zone,
        # если для обоих типов классификации используются одинаковые модели,
        # то указатели на такую модель могут быть в обоихк ключах
        self.model_index = dict()

    # назначет соответствующие модели в индекс
    # model_type может быть только: svm - метод опорных веторов
    #                               knn - метод k-ближайших соседей
    #                               nbc - наивный Байесовский классификатор
    # variable может быть только  зона, наименование и оба
    # wine_type - только white, или red
    def __assign_model(self, wine_type, dep_variable, model_type, model_object):
        if dep_variable == 'зона':
            self.model_index['{}_{}_zone'.format(wine_type, model_type)] = model_object
        elif dep_variable == 'наименование':
            self.model_index['{}_{}_type'.format(wine_type, model_type)] = model_object
        elif dep_variable == 'оба':
            self.model_index['{}_{}_zone'.format(wine_type, model_type)] = model_object
            self.model_index['{}_{}_type'.format(wine_type, model_type)] = model_object
        else:
            # TODO: сюда запихать обработчик ошибок
            pass

    # должен инициализироваться при запуске приложения
    def load_models(self, model_file):
        # должен из указанного файла внести в объект
        # TODO: сделать нормальный ввод
        # в примере взяты модели из работы
        overall_data = {'white_knn': {'dependents': ['оба'],
                                      'neighbors_amount': [1],
                                      'metric': ['euclidean'],
                                      'weights': ['uniform']},
                        'white_svm': {'dependents': ['зона', 'наименование'],
                                      'type': ['C', 'C'],
                                      'param': [2, 12],
                                      'kernel': ['poly', 'rbf'],
                                      'k_params': [{'degree': 3, 'gamma': 1, 'coefficient': 1}, {'gamma': 0.067}]},
                        'red_knn': {'dependents': ['оба'],
                                    'neighbors_amount': [1],
                                    'metric': ['euclidean'],
                                    'weights': ['uniform']},
                        'red_svm': {'dependents': ['зона', 'наименование'],
                                    'type': ['C', 'Nu'],
                                    'param': [10, 0.2],
                                    'kernel': ['rbf', 'rbf'],
                                    'k_params': [{'gamma': 0.067}, {'gamma': 0.067}]}
                        }
        # белые svm
        data_dict = overall_data['white_svm']
        models = self.__create_svm(data_dict)
        for variable, model in models.items():
            self.__assign_model('white', variable, 'svm', model)
        # красные svm
        data_dict = overall_data['red_svm']
        models = self.__create_svm(data_dict)
        for variable, model in models.items():
            self.__assign_model('red', variable, 'svm', model)
        # белые knn
        data_dict = overall_data['white_knn']
        models = self.__create_knn(data_dict)
        for variable, model in models.items():
            self.__assign_model('white', variable, 'knn', model)
        # красные knn
        data_dict = overall_data['red_knn']
        models = self.__create_knn(data_dict)
        for variable, model in models.items():
            self.__assign_model('red', variable, 'knn', model)
        # белые nbc
        models = self.__create_nbc()
        for variable, model in models.items():
            self.__assign_model('white', variable, 'nbc', model)
        models = self.__create_nbc()
        for variable, model in models.items():
            self.__assign_model('red', variable, 'nbc', model)

    # создает объект классификационной модели метода опорных векторов
    # model_data - словарь с необходимыми данными для постройки модели
    @staticmethod
    def __create_svm(data_dict):
        result_models = dict()
        for model_num, variable in enumerate(data_dict['dependents']):
            m_type = data_dict['type'][model_num]
            param = data_dict['param'][model_num]
            kernel = data_dict['kernel'][model_num]
            kernel_dict = dict()
            for key_name, key_val in data_dict['k_params'][model_num].items():
                kernel_dict[key_name] = key_val
            # дозаполнение данных
            # degree
            if 'degree' not in kernel_dict.keys():
                kernel_degree = 3
            else:
                kernel_degree = kernel_dict['degree']
            # gamma
            if 'gamma' not in kernel_dict.keys():
                kernel_gamma = 0.067
            else:
                kernel_gamma = kernel_dict['gamma']
            # coefficient
            if 'coefficient' not in kernel_dict.keys():
                kernel_coefficient = 1
            else:
                kernel_coefficient = kernel_dict['coefficient']
            # TODO: сделать взыов и обработку ошибок на использование некорректных данных
            # создание модели
            if m_type == 'C':
                result_models[variable] = SVC(param,
                                              kernel,
                                              kernel_degree,
                                              kernel_gamma,
                                              kernel_coefficient)
            elif m_type == 'Nu':
                result_models[variable] = NuSVC(param,
                                                kernel,
                                                kernel_degree,
                                                kernel_gamma,
                                                kernel_coefficient)
            else:
                # TODO: добавить исключение
                pass
        return result_models

    @staticmethod
    def __create_knn(data_dict):
        result_models = dict()
        for model_num, dep_variable in enumerate(data_dict['dependents']):
            n_amount = data_dict['neighbors_amount'][model_num]
            metric = data_dict['metric'][model_num]
            weights = data_dict['weights'][model_num]
            model = KNeighborsClassifier(n_amount, weights, metric=metric)
            result_models[dep_variable] = model
        return result_models

    @staticmethod
    def __create_nbc():
        result_models = {'оба': GaussianNB()}
        return result_models


if __name__ == '__main__':
    test_models = Models()
    test_models.load_models(None)
    for model_type, model in test_models.model_index.items():
        print(model_type, model)
