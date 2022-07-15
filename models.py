from abc import ABC
import fasttext
import jaro
import pandas as pd
import textdistance as td
from matplotlib import pyplot
from plugin import ExternalPlugin
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess


# Creación de la clase Padre
class Model(ABC):
    # Inicialización de los parámetros
    def __init__(self, **kwargs):
        self.nombre = kwargs["metodo"]
        self.param1 = kwargs["param1"]
        self.operacion = kwargs["operacion"]
        self.df = kwargs["dataframe"]

    def predecir(self):
        pass

    def entrenar(self):
        pass

    # Realización de cálculos: dependiendo del parámetro, serán resultados gráficos o un Dataframe
    def calculos(self, resultados):
        if self.operacion == 'blocking':
            pyplot.plot(resultados[0], resultados[1], label='Recall')
            pyplot.plot(resultados[0], resultados[2], label='Precision')
            pyplot.plot(resultados[0], resultados[3], label='F1 Score')
            pyplot.axvline(x=0.05, color="red")
            pyplot.axvline(x=0.15, color="red")

            # Etiquetas de los ejes
            pyplot.xlabel('Valor del Parámetro')
            pyplot.ylabel('KPI seleccionado')
            pyplot.legend()
            return pyplot
        elif self.operacion == 'matching':
            pyplot.plot(resultados[0], resultados[1], label='Recall')
            pyplot.plot(resultados[0], resultados[2], label='Precision')
            pyplot.plot(resultados[0], resultados[3], label='F1 Score')
            pyplot.axvline(x=0.85, color="red")  # Plotting a single vertical line
            pyplot.axvline(x=0.95, color="red")  # Plotting a single vertical line

            # Etiquetas de los ejes
            pyplot.xlabel('Valor del Parámetro')
            pyplot.ylabel('KPI seleccionado')
            pyplot.legend()
            return pyplot
        elif self.operacion == 'calcular':
            is_1 = resultados[1]['label'] == 1
            rechazados_1 = resultados[1][is_1]

            is_1 = resultados[0]['label'] == 1
            aceptados_1 = resultados[0][is_1]

            tp = len(aceptados_1)
            fp = len(resultados[0]) - len(aceptados_1)
            fn = len(rechazados_1)
            tn = len(resultados[1]) - len(rechazados_1)
            texto_resultado = ''
            texto_resultado += 'TP: ' + str(tp) + ' FP: ' + str(fp) + ' FN: ' + str(fn) + ' TN: ' + str(tn)
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            if (tp + fp) == 0:
                precision = 1.0
            else:
                precision = tp / (tp + fp)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            texto_resultado2 = 'RECALL: ' + str(recall) + ' PRECISION: ' + str(precision) + ' F1-SCORE:' + str(
                f1_score)

            return resultados[0], texto_resultado, texto_resultado2


# Desarrollo de los modelos TextDistance
class TextDistance(Model):
    def predecir(self):
        aceptados = []
        rechazados = []
        pred_todo = []
        for index, row in self.df.iterrows():
            if getattr(self, 'nombre') == "Jaccard":
                resultadoTodas = self.__jaccard(row)
            elif getattr(self, 'nombre') == "Jaro Winkler":
                resultadoTodas = self.__jaro_winkler(row)
            elif getattr(self, 'nombre') == "Hamming":
                resultadoTodas = self.__hamming(row)
            pred_todo.append(resultadoTodas)
            if resultadoTodas > self.param1:
                aceptados.append(row)
            else:
                rechazados.append(row)
        aceptados_df, rechazados_df = self.__convertirADF(aceptados, rechazados)
        resultados = aceptados_df, rechazados_df, pred_todo
        if self.operacion == 'blocking':
            return resultados
        if self.operacion == 'matching':
            return resultados
        if self.operacion == 'calcular':
            return self.calculos(resultados)

    def blocking(self):
        opciones = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
        recallJ = []
        precisionJ = []
        f1_scoreJ = []

        for opcion in opciones:
            labels = list(self.df["label"])
            self.param1 = opcion
            data_blocking_df, rechazados_df, pred_todo = self.predecir()
            is_1 = rechazados_df['label'] == 1
            rechazados_1 = rechazados_df[is_1]

            is_1 = data_blocking_df['label'] == 1
            aceptados_1 = data_blocking_df[is_1]
            tp = len(aceptados_1)
            fp = len(data_blocking_df) - len(aceptados_1)
            fn = len(rechazados_1)
            tn = len(rechazados_df) - len(rechazados_1)
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            if (tp + fp) == 0:
                precision = 1.0
            else:
                precision = tp / (tp + fp)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            recallJ.append(recall)
            precisionJ.append(precision)
            f1_scoreJ.append(f1_score)
        resultados = opciones, recallJ, precisionJ, f1_scoreJ
        self.calculos(resultados)

    def __convertirADF(self, aceptados, rechazados):
        aceptados_df = pd.DataFrame(aceptados,
                                    columns=['left_title', 'left_description', 'left_brand', 'left_specTableContent',
                                             'right_title', 'right_description', 'right_brand',
                                             'right_specTableContent', 'label', 'id'])
        rechazados_df = pd.DataFrame(rechazados,
                                     columns=['left_title', 'left_description', 'left_brand', 'left_specTableContent',
                                              'right_', 'right_description', 'right_brand',
                                              'right_specTableContent', 'label', 'id'])
        return aceptados_df, rechazados_df

    def __jaccard_similarity(self, query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection) / len(union)

    def __jaro_winkler(self, row):
        return jaro.jaro_winkler_metric(row['all_left'], row['all_right'])

    def __jaccard(self, row):
        tokenize = lambda doc: doc.lower().split(" ")
        return self.__jaccard_similarity(tokenize(row['all_left']), tokenize(row['all_right']))

    def __hamming(self, row):
        tokenize = lambda doc: doc.lower().split(" ")
        return td.hamming.normalized_similarity(tokenize(row['all_left']), tokenize(row['all_right']))


# Desarrollo del modelo FastText
class FastText(Model):
    def entrenar(self):
        documentos = []
        for index, row in self.df.iterrows():
            documentos.append(row['all_left'])
            documentos.append(row['all_right'])
        fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

        dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documentos])

        similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                                nonzero_limit=100)

        self.predecir(dictionary, similarity_matrix)

    def predecir(self, dictionary, similarity_matrix):
        aceptados = []
        rechazados = []
        pred_todo = []
        pred_titulo = []
        for index, row in self.df.iterrows():
            resultadoTitulo = self.__soft_cosine_similarity(dictionary.doc2bow(simple_preprocess(row['all_left'])),
                                                            dictionary.doc2bow(simple_preprocess(row['all_right'])),
                                                            similarity_matrix)
            pred_titulo.append(resultadoTitulo)
            if (resultadoTitulo > 0.65):
                aceptados.append(row)
            else:
                rechazados.append(row)
        aceptados_df, rechazados_df = self.__convertirADF(aceptados, rechazados)
        resultados = aceptados_df, rechazados_df, pred_todo
        return resultados

    def blocking(self):
        opciones = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
        recallJ = []
        precisionJ = []
        f1_scoreJ = []

        for opcion in opciones:
            labels = list(self.df["label"])
            self.param1 = opcion
            data_blocking_df, rechazados_df, pred_todo = self.entrenar()
            is_1 = rechazados_df['label'] == 1
            rechazados_1 = rechazados_df[is_1]

            is_1 = data_blocking_df['label'] == 1
            aceptados_1 = data_blocking_df[is_1]
            tp = len(aceptados_1)
            fp = len(data_blocking_df) - len(aceptados_1)
            fn = len(rechazados_1)
            tn = len(rechazados_df) - len(rechazados_1)
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            if (tp + fp) == 0:
                precision = 1.0
            else:
                precision = tp / (tp + fp)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

            recallJ.append(recall)
            precisionJ.append(precision)
            f1_scoreJ.append(f1_score)
        resultados = opciones, recallJ, precisionJ, f1_scoreJ
        self.calculos(resultados)

    def __soft_cosine_similarity(self, sent_1, sent_2, similarity_matrix):
        return softcossim(sent_1, sent_2, similarity_matrix)

    def __convertirADF(self, aceptados, rechazados):
        aceptados_df = pd.DataFrame(aceptados,
                                    columns=['title_left', 'description_left', 'brand_left', 'specTableContent_left',
                                             'title_right', 'description_right', 'brand_right',
                                             'specTableContent_right', 'label', 'pair_id', 'all_right',
                                             'all_left'])
        rechazados_df = pd.DataFrame(rechazados,
                                     columns=['title_left', 'description_left', 'brand_left', 'specTableContent_left',
                                              'title_right', 'description_right', 'brand_right',
                                              'specTableContent_right', 'label', 'pair_id', 'all_right',
                                              'all_left'])
        return aceptados_df, rechazados_df

    def __modelo_manual(self, nombre, modelo, d_lr, d_dim, d_ws, d_epoch):
        return fasttext.train_unsupervised(nombre, model=modelo, lr=d_lr, dim=d_dim, ws=d_ws, epoch=d_epoch)


# Desarrollo del Plugin para modelos externos
class Plugin(Model):
    def __init__(self, **kwargs):
        self.nombre = kwargs["metodo"]
        self.predicciones = []

    def predecir(self):
        ExternalPlugin.predecir(self, modelo)

    def entrenar(self):
        self.predicciones = ExternalPlugin.entrenar2(self)

    def calcular(self):
        ExternalPlugin.calcular(self, self.predicciones)
