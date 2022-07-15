from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib import pyplot

class ExternalPlugin:
    def entrenar(self):
        model = SentenceTransformer('all-MiniLM-L6-v2')

        entrenamiento = pd.read_csv('ent_red.csv')
        entrenamiento = entrenamiento.astype({"label": float}, errors='raise')
        train_ex = []

        for index, row in entrenamiento.iterrows():
            t_all_left = str(row.left_title) + ' ' + str(row.left_description) + ' ' + str(row.left_brand) + ' ' + \
                         str(row.left_specTableContent)
            t_all_right = str(row.right_title) + ' ' + str(row.right_description) + ' ' + str(row.right_brand) + ' ' + \
                          str(row.right_specTableContent)
            train_ex.append(InputExample(texts=[t_all_left, t_all_right], label=row['label']))

        resultados = []
        test = pd.read_csv('test_red.csv')
        for index, row in test.iterrows():
            cos_sim = util.cos_sim(model.encode(row['left_title']), model.encode(row['right_title']))
            resultados.append(cos_sim)
        return (resultados)
    def entrenar2(self):
        print("x")
        # Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        entrenamiento = pd.read_csv('ent_red.csv')
        validacion = pd.read_csv('val_red.csv')
        entrenamiento = entrenamiento.astype({"label": float}, errors='raise')
        validacion = validacion.astype({"label": float}, errors='raise')
        train_ex = []
        for index, row in entrenamiento.iterrows():
            t_all_left = str(row.left_title) + ' ' + str(row.left_description) + ' ' + str(row.left_brand) + ' ' + \
                         str(row.left_specTableContent)
            t_all_right = str(row.right_title) + ' ' + str(row.right_description) + ' ' + str(row.right_brand) + ' ' + \
                          str(row.right_specTableContent)
            train_ex.append(InputExample(texts=[t_all_left, t_all_right], label=row['label']))

        sentences1 = []
        sentences2 = []
        for index, row in validacion.iterrows():
            v_all_left = str(row.left_title) + ' ' + str(row.left_description) + ' ' + str(row.left_brand) + ' ' + \
                         str(row.left_specTableContent)
            v_all_right = str(row.right_title) + ' ' + str(row.right_description) + ' ' + str(row.right_brand) + ' ' + \
                          str(row.right_specTableContent)
            sentences1.append(v_all_left)
            sentences2.append(v_all_right)
        scores = validacion['label']

        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

        # Define your train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(train_ex, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)

        # Tune the model
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100,
                  evaluator=evaluator, evaluation_steps=150)
        # return model

        resultados = []
        test = pd.read_csv('test_red.csv')
        for index, row in test.iterrows():
            test_all_left = str(row.left_title) + ' ' + str(row.left_description) + ' ' + str(row.left_brand) + ' ' + \
                         str(row.left_specTableContent)
            test_all_right = str(row.right_title) + ' ' + str(row.right_description) + ' ' + str(row.right_brand) + ' ' + \
                          str(row.right_specTableContent)
            cos_sim = util.cos_sim(model.encode(test_all_left), model.encode(test_all_right))
            resultados.append(cos_sim)
        return (resultados)

    def predecir(self, modelo):
        resultados = []
        test = pd.read_csv('test.csv')
        for index, row in test.iterrows():
            cos_sim = util.cos_sim(modelo.encode(row['left_title']), modelo.encode(row['right_title']))
            resultados.append(cos_sim)
        return (resultados)

    def calcular(self, predicciones):
        print("a")
        test = pd.read_csv('test_red.csv')
        label = test['label']
        opciones = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]
        recallJ = []
        precisionJ = []
        f1_scoreJ = []
        kpis = []
        todo = []
        for opcion in opciones:
            contador = 0
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            resultados = []
            for p in predicciones:
                if p > opcion:
                    resultados.append(1)
                else:
                    resultados.append(0)
            for i in label:
                if i == 1:
                    if resultados[contador] == 1:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                if i == 0:
                    if resultados[contador] == 1:
                        fp = fp + 1
                    else:
                        tn = tn + 1
                contador = contador + 1
            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            if (tp + fp) == 0:
                print("uo")
                precision = 1.0
            else:
                precision = tp / (tp + fp)
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            todos = [tp, fp, tn, fn]
            todo.append(todos)
            recallJ.append(recall)
            precisionJ.append(precision)
            f1_scoreJ.append(f1_score)

        pyplot.plot(opciones, recallJ, label='Recall')
        pyplot.plot(opciones, precisionJ, label='Precision')
        pyplot.plot(opciones, f1_scoreJ, label='F1 Score')
        pyplot.axvline(x=0.85, color="red")  # Plotting a single vertical line
        pyplot.axvline(x=0.95, color="red")  # Plotting a single vertical line

        # Etiquetas de los ejes
        pyplot.xlabel('Valor del Parámetro')
        pyplot.ylabel('KPI seleccionado')
        pyplot.legend()
        pyplot.show()

        print(predicciones)
        print(label)
