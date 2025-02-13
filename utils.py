def incremental_training(dataset, modelo):
    acc_values = []
    
    acc = metrics.Accuracy()
    report = metrics.ClassificationReport()
    counter = 0

    model = compose.Pipeline(preprocessing.StandardScaler(), modelo())

    for x, y in dataset:
        y_pred = model.predict_one(x)
        model.learn_one(x,y)
        if y_pred != None:
            report.update(y,y_pred)
            acc.update(y,y_pred)
        if counter % 25 == 0:
            print(f'{acc}')
            acc_values.append(acc.get())
        counter+=1

    return acc_values, report

def incremental_training_concept_drift_detector(dataset, modelo, drift_detect):
    # Inicializamos los detectores 
    drift_detector = drift_detect()
    drift_detector2 = drift_detect()
    # Almacenamos los indices donde hay cambios
    drifts = []
    drifts2 = []
    # Almacenamos los valores de accuracy segun vayan llegando datos.
    acc_values = []

    model = compose.Pipeline(preprocessing.StandardScaler(), modelo())

    acc2 = metrics.Accuracy()
    report2 = metrics.ClassificationReport()
    counter = 0

    for i, (x_dict, y) in enumerate(dataset):
        x = list(x_dict.values())

        drift_detector.update(x[0])
        drift_detector2.update(x[1])

        if drift_detector.drift_detected or drift_detector2.drift_detected:
            print(f'Change detected at index {i}')
            # Guardar el indice donde se detecta un cambio.
            if drift_detector.drift_detected:
                drifts.append(i)
            else:
                drifts2.append(i)

            # Reiniciar el modelo cuando se detecta un cambio
            model = compose.Pipeline(preprocessing.StandardScaler(), modelo())

        y_pred = model.predict_one(x_dict)
        model = model.learn_one(x_dict, y)
        
        if y_pred != None:
            report2.update(y, y_pred)
            acc2.update(y, y_pred)

        if counter % 25 == 0:
            print(f'{acc2}')
            acc_values.append(acc2.get())
        counter += 1

    return acc_values, drifts, drifts2, report2

def plot_accuracy_dual(acc_values_1, acc_values_2, drifts, drifts2):
    """
    Dibuja la precisión del modelo con y sin detector segun van obteniendo más datos,
    permitiendo visualizar el efecto del detector.

    Parámetros:
    - acc_values_1: Lista con los valores de precisión para el modelo incremental sin detector.
    - acc_values_2: Lista con los valores de precisión para el modelo incremental con detector.
    - drifts: Lista con los indices donde se produjeron cambios de la distribución de x1.
    - drifts2: Lista con los indices donde se produjeron cambios de la distribución de x2.

    """
    indices = np.arange(min(len(acc_values_1), len(acc_values_2)))

    indices = indices*25

    plt.plot(indices, acc_values_1, label='Accuracy sin detector', color='blue')
    plt.plot(indices, acc_values_2, label='Accuracy con detector', color='orange')
    for value in drifts:
        plt.axvline(x=value, color='red', linestyle='--', label='Drift 1')
    for value2 in drifts2:
        plt.axvline(x=value2, color='green', linestyle='--', label='Drift 2')

    custom_legend = [
        plt.Line2D([0], [0], color='blue', label='Accuracy sin detector'),
        plt.Line2D([0], [0], color='orange', label='Accuracy con detector'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Drift 1 (rojo)'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='Drift 2 (verde)')
    ]

    plt.legend(handles=custom_legend)
    plt.xlabel('Índices')
    plt.ylabel('Precisión')
    plt.title('Precisión a lo largo del tiempo')
    plt.show()