import pickle

import utilities


class Analyzer():
    """Analyze the saved results."""

    def __init__(self):
        """Nothing to do."""

        pass

    def experiment_analysis1(self, folder="", max_attr_numb=85):
        """Analyse the results of the first experiment."""

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "absolute_results.pickle", "rb") as act:
            result_dict = pickle.load(act)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "relative_results.pickle", "rb") as act:
            relative_results = pickle.load(act)
        act.close()
        max_attr_numb += 1
        relevant = relative_results["total_result"]
        x_values = [i for i in range(1, max_attr_numb)]
        y_values = [relevant[i][0] for i in range(1, max_attr_numb)]
        utilities.create_line_plot(x_values, y_values, x_name="top-x predictions", y_name="Percentage of correct predicted attributes",  title="Percentage of correct attributes of top-x predicted attribtues",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "percentage_correct_attribtues.png")
        y_values = [relevant[i][1] for i in range(1, max_attr_numb)]
        utilities.create_line_plot(x_values, y_values, x_name="top-x predictions", y_name="Nubmer of correct predicted attribtues", title="Number of correct attributes of top-x predicted attribtues",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "number_correct_attribtues.png")
        y_values = [round(relevant["percentage_distribution"][i].item(), 4)
                    for i in range(len(relevant["percentage_distribution"]))]

        utilities.create_bar_plot(y_values, x_name="top predictions", y_name="Average probability prediction", title="Number of correct attributes of top-x predicted attribtues",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "probability_mass.png")
        utilities.create_bar_plot(y_values[:5], x_name="top predictions", y_name="Average probability prediction", title="Number of correct attributes of top-x predicted attribtues",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "probability_mass_top5.png")
        utilities.create_bar_plot(y_values[:10], x_name="top predictions", y_name="Average probability prediction", title="Number of correct attributes of top-x predicted attribtues",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "probability_mass_top10.png")

    def experiment1_analysis_per_attr(self, attr_idx_label):
        """Analysis the results of experiment1 per attribute."""

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/experiment1_per_attr_relative_results.pickle", "rb") as act:
            result_dict = pickle.load(act)
        act.close()
        x_val = [attr_idx_label[i] for i in range(1, 86)]  # Attribute names.
        y_val = [result_dict[i]["accuracy"] for i in range(1, 86)]
        x_values_sorted = [x for x, y in sorted(zip(x_val, y_val), key=lambda pair: pair[1])]
        y_values_sorted = [y for x, y in sorted(zip(x_val, y_val), key=lambda pair: pair[1])]
        utilities.create_bar_plot(y_values_sorted, x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/attr_accuracy.png", x_values=x_values_sorted)
        utilities.create_bar_plot(y_values_sorted[86-10:], x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/attr_accuracy_topx.png", x_values=x_values_sorted[86-10:])
        utilities.create_bar_plot(y_values_sorted[:10], x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/first_experiment/attr_accuracy_lowx.png", x_values=x_values_sorted[:10])

    def experiment_analysis2(self, folder=""):
        """Analyse the results of the first experiment."""

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "absolute_results.pickle", "rb") as act:
            result_dict = pickle.load(act)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "relative_results.pickle", "rb") as act:
            relative_results = pickle.load(act)
        act.close()
        relevant = relative_results["total_result"]
        x_values = [i for i in range(0, 101, 1)]
        y_values = [relevant[i][0].item() for i in range(0, 101, 1)]
        print(y_values)
        utilities.create_line_plot(x_values, y_values, x_name="top-% predictions", y_name="Actual percentage used",  title="Minimum top-x percentage that is higher than the minimum percentage",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "actual_percentage.png")
        y_values = [relevant[i][1] for i in range(0, 101, 1)]
        utilities.create_line_plot(x_values, y_values, x_name="top-% predictions", y_name="Average number of attributes found", title="Average number of attributes found for given %",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "average_found_attributes.png")
        y_values = [relevant[i][2] for i in range(0, 101, 1)]
        utilities.create_line_plot(x_values, y_values, x_name="top-% predictions", y_name="Average number of correct attributes found", title="Average number of correct attributes found for given %",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "average_found_correct_attributes.png")
        y_values = [relevant[i][3] for i in range(0, 101, 1)]
        utilities.create_line_plot(x_values, y_values, x_name="top-% predictions", y_name="Average % of correct attributes", title="Average % of correct predicted attribtues.",
                                   save_path="/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "average_percent_correct_attributes.png")

    def experiment_analysis_category3(self, folder: str, category: str, used_attr_idxs: list, attr_idx_label_mapping: dict, avg_prob=False):
        """Analyze the experimental results of the category analysis."""

        with open("/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "absolute_results.pickle", "rb") as act:
            result_dict = pickle.load(act)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "relative_results.pickle", "rb") as act:
            relative_results = pickle.load(act)
        act.close()
        relevant = relative_results["total_result"]
        x_values = [i for i in range(1, len(relevant.keys()) - 1)]
        y_values = [relevant[i][0] for i in x_values]

        if avg_prob:
            x_1 = x_values.copy()
            y_1 = [avg_prob for i in x_1]
            utilities.create_line_plot(x_values, y_values, x_name="top predictions", y_name="Accuracy",  title="Accuracy of top predictions for " + category,
                                       save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "average_percent_correct_attributes.png",
                                       x_values1=x_1, y_values1=y_1)
        else:
            utilities.create_line_plot(x_values, y_values, x_name="top predictions", y_name="Accuracy",  title="Accuracy of top predictions for " + category,
                                       save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "average_percent_correct_attributes.png")

        y_values = [round(relevant["percentage_distribution"][i].item(), 4)
                    for i in range(len(relevant["percentage_distribution"]))]
        utilities.create_bar_plot(y_values, x_name="top predictions", y_name="Average probability prediction", title="Percentage distribution",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "probability_mass.png")
        utilities.create_bar_plot(y_values[:5], x_name="top predictions", y_name="Average probability prediction", title="Percentage distribution",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "probability_mass_top5.png")
        utilities.create_bar_plot(y_values[:10], x_name="top predictions", y_name="Average probability prediction", title="Percentage distribution",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "probability_mass_top10.png")

        x_values = [attr_idx_label_mapping[i] for i in used_attr_idxs]
        y_values = [relative_results[i]["top-1-accuracy"] for i in used_attr_idxs]
        numb = relative_results["total_result"]["count"]
        z_values = [relative_results[i]["count"] / numb for i in used_attr_idxs]
        x_values_sorted = [x for x, y, z in sorted(zip(x_values, y_values, z_values), key=lambda pair: pair[1])]
        y_values_sorted = [y for x, y, z in sorted(zip(x_values, y_values, z_values), key=lambda pair: pair[1])]
        z_values_sorted = [z for x, y, z in sorted(zip(x_values, y_values, z_values), key=lambda pair: pair[1])]
        utilities.create_bar_plot(y_values_sorted, x_name="top predictions", y_name="Accuracy", title="Accuracy of top predictions",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "attr_performance.png", x_values=x_values_sorted)

        utilities.create_bar_plot(z_values_sorted, x_name="top predictions", y_name="Part of top-1 prediction", title="Percentage of top predictions",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "attr_prediction.png", x_values=x_values_sorted)

    def experiment_analysis_4(self, folder: str, attr_idx_label_mapping: dict, topx=False):
        """Analyze the experimental results of the category analysis."""

        with open("/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "absolute_results.pickle", "rb") as act:
            result_dict = pickle.load(act)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "relative_results.pickle", "rb") as act:
            relative_results = pickle.load(act)
        act.close()
        acc = [(relative_results[key]["true_positive"] + relative_results[key]["true_negative"]) /
               (relative_results[key]["true_positive"] + relative_results[key]["false_negative"] + relative_results[key]["false_positive"] + relative_results[key]["true_negative"]) for key in range(1, 86)]
        rec = [relative_results[key]["true_positive"] /
               (relative_results[key]["true_positive"] + relative_results[key]["false_negative"]) for key in range(1, 86)]
        prec = [relative_results[key]["true_positive"] /
                (relative_results[key]["true_positive"] + relative_results[key]["false_positive"]) for key in range(1, 86)]
        part = [(relative_results[key]["true_positive"] + relative_results[key]["false_negative"]) /
                (relative_results[key]["true_positive"] + relative_results[key]["false_negative"] + relative_results[key]["false_positive"] + relative_results[key]["true_negative"]) for key in range(1, 86)]
        f1 = [2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(len(acc))]
        labels = [attr_idx_label_mapping[key] for key in range(1, 86)]
        f1_sorted = [k for x, y, z, b, k, t in sorted(zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]
        acc_sorted = [x for x, y, z, b, k, t in sorted(zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]
        rec_sorted = [y for x, y, z, b, k, t in sorted(zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]
        prec_sorted = [z for x, y, z, b, k, t in sorted(
            zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]
        labels_sorted = [b for x, y, z, b, k, t in sorted(
            zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]
        part_sorted = [t for x, y, z, b, k, t in sorted(
            zip(acc, rec, prec, labels, f1, part), key=lambda pair: pair[1])]

        if topx:
            utilities.create_bar_plot(part_sorted[:topx], x_name="Attributes", y_name="Percentage", title="Percentage of samples with that attribtue",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "perc_of_samples_lowx.png", x_values=labels_sorted[:topx])
            utilities.create_bar_plot(part_sorted[len(acc_sorted) - topx:], x_name="Attributes", y_name="f1-score", title="Percentage of samples with that attribtue",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "perc_of_samples_topx.png", x_values=labels_sorted[len(acc_sorted) - topx:])

        utilities.create_bar_plot(part_sorted, x_name="Attributes", y_name="Percentage", title="Percentage of samples with that attribtue",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "perc_of_samples.png", x_values=labels_sorted)

        if topx:
            utilities.create_bar_plot(f1_sorted[:topx], x_name="Attributes", y_name="f1-score", title="f1-score per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "f1_attr_prediction_lowx.png", x_values=labels_sorted[:topx])
            utilities.create_bar_plot(f1_sorted[len(acc_sorted) - topx:], x_name="Attributes", y_name="f1-score", title="f1-score per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "f1_attr_prediction_topx.png", x_values=labels_sorted[len(acc_sorted) - topx:])

        utilities.create_bar_plot(f1_sorted, x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "accuracy_attr_prediction.png", x_values=labels_sorted)

        if topx:
            utilities.create_bar_plot(acc_sorted[:topx], x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "accuracy_attr_prediction_lowx.png", x_values=labels_sorted[:topx])
            utilities.create_bar_plot(acc_sorted[len(acc_sorted) - topx:], x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "accuracy_attr_prediction_topx.png", x_values=labels_sorted[len(acc_sorted) - topx:])

        utilities.create_bar_plot(acc_sorted, x_name="Attributes", y_name="Accuracy", title="Accuracy per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "accuracy_attr_prediction.png", x_values=labels_sorted)

        if topx:
            utilities.create_bar_plot(prec_sorted[:topx], x_name="Attributes", y_name="Precision", title="Precision per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "precision_attr_prediction_lowx.png", x_values=labels_sorted[:topx])
            utilities.create_bar_plot(prec_sorted[len(acc_sorted) - topx:], x_name="Precision", y_name="Precision", title="Precision per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "precision_attr_prediction_topx.png", x_values=labels_sorted[len(acc_sorted) - topx:])

        utilities.create_bar_plot(prec_sorted, x_name="Attributes", y_name="Precision", title="Precision per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "precision_attr_prediction.png", x_values=labels_sorted)

        if topx:
            utilities.create_bar_plot(rec_sorted[:topx], x_name="Attributes", y_name="Recall", title="Recall per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "recall_attr_prediction_lowx.png", x_values=labels_sorted[:topx])
            utilities.create_bar_plot(rec_sorted[len(acc_sorted) - topx:], x_name="Recall", y_name="Recall", title="Recall per attribute",
                                      save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "recall_attr_prediction_topx.png", x_values=labels_sorted[len(acc_sorted) - topx:])

        utilities.create_bar_plot(rec_sorted, x_name="Attributes", y_name="Recall", title="Recall per attribute",
                                  save_path="/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "recall_attr_prediction.png", x_values=labels_sorted)

    def check_if_class_has_attribute(self, small_list, class_dict, class_weights: dict):
        """Check how many classes have at least one attribute of a list in their label attribute list."""

        cnt = 0
        found_attr = 0
        found_count = 0
        small_list = set(small_list)
        for class_idx in class_dict.keys():
            cnt += 1
            attr_list = set(class_dict[class_idx])
            intersect = len(attr_list.intersection(small_list))
            found_attr += (intersect > 0) * class_weights["class"][class_idx]
            found_count += intersect * class_weights["class"][class_idx]
        result = {"average_occurrence": found_count / class_weights["samples"], "percentage_found": found_attr /
                  class_weights["samples"], "numb_found": found_count, "numb_classes": cnt}
        return result

    def plot_occurrences(self, topx=85, lowx=False):
        """Plot the occurrences of each attribute."""

        with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/attr_readable_occurence.pickle", "rb") as act:
            res_mapping_readable = pickle.load(act)
        act.close()
        save_path = "/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/attr_occurrences_plot_top" + \
            str(topx) + ".png"
        x_values = []
        y_values = []
        for key in res_mapping_readable.keys():
            x_values.append(key)
            y_values.append(res_mapping_readable[key])
        x_values_sorted = [x for x, y in sorted(zip(x_values, y_values), key=lambda pair: pair[1])]
        y_values_sorted = [y for x, y in sorted(zip(x_values, y_values), key=lambda pair: pair[1])]

        utilities.create_bar_plot(y_values=y_values_sorted[85 - topx:], title="Occurrences of attributes", save_path=save_path,
                                  x_name="Attributes", y_name="Occurrences", x_values=x_values_sorted[85 - topx:])
        if lowx:
            save_path = "/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/attr_occurrences_plot_low" + \
                str(lowx) + ".png"
            utilities.create_bar_plot(y_values=y_values_sorted[:lowx], title="Occurrences of attributes", save_path=save_path,
                                      x_name="Attributes", y_name="Occurrences", x_values=x_values_sorted[:lowx])

    def plot_new_occurrences(self, topx=85, lowx=False):
        """Plot the occurrences of each attribute."""

        with open("/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_readable_occurence.pickle", "rb") as act:
            res_mapping_readable = pickle.load(act)
        act.close()
        save_path = "/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_occurrences_plot.png"
        x_values = []
        y_values = []
        for key in res_mapping_readable.keys():
            x_values.append(key)
            y_values.append(res_mapping_readable[key])
        x_values_sorted = [x for x, y in sorted(zip(x_values, y_values), key=lambda pair: pair[1])]
        y_values_sorted = [y for x, y in sorted(zip(x_values, y_values), key=lambda pair: pair[1])]

        utilities.create_bar_plot(y_values=y_values_sorted, title="Occurrences of attributes", save_path=save_path,
                                  x_name="Attributes", y_name="Occurrences", x_values=x_values_sorted)
        save_path = "/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_occurrences_plot_top" + \
            str(topx) + ".png"
        utilities.create_bar_plot(y_values=y_values_sorted[85 - topx:], title="Occurrences of attributes", save_path=save_path,
                                  x_name="Attributes", y_name="Occurrences", x_values=x_values_sorted[85 - topx:])
        if lowx:
            save_path = "/home/felix/new_bachelor/awa2/dataset_analysis/laion400M_search/new_attr_occurrences_plot_low" + \
                str(lowx) + ".png"
            utilities.create_bar_plot(y_values=y_values_sorted[:lowx], title="Occurrences of attributes", save_path=save_path,
                                      x_name="Attributes", y_name="Occurrences", x_values=x_values_sorted[:lowx])

    def analyze_class_prediction(self, name_beginning="", metric="recall", max_numb=23):
        """Plot the accuracy of class prediction depending on the number of attributes."""

        x_val = []
        y_val = []
        for numb in range(0, max_numb):
            folder = name_beginning + str(numb) + "_"
            print("Folder: ", folder)
            with open("/home/felix/new_bachelor/awa2/results/predict_class/" + folder + "report_idx.pickle", "rb") as act:
                result = pickle.load(act)
            act.close()
            x_val.append(numb)
            y_val.append(result["weighted avg"][metric])
        utilities.create_line_plot(x_val, y_val, x_name="Number of attributes", y_name=metric,  title="Performance depending on number of attributes",
                                   save_path="/home/felix/new_bachelor/awa2/results/predict_class/attr_numb_" + name_beginning[:-1] + ".png")

    def category_peformance_against_random_guessing(self, folder: str, random_acc: float, new_categories=False):
        """Calculate for each category how well the model performed against random guessing for the top-1 prediction."""

        if new_categories:
            with open("/home/felix/new_bachelor/awa2/results/predict_category/new_categories/" + folder + "relative_results.pickle", "rb") as act:
                relative_results = pickle.load(act)
            act.close()

        else:
            with open("/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "relative_results.pickle", "rb") as act:
                relative_results = pickle.load(act)
            act.close()
        relevant = relative_results["total_result"]
        pred_acc = relevant[1][0]
        print(folder[:-1], ": Predicting accuracy / random guessing accuracy: ", pred_acc /
              random_acc, " absoulute percentage difference: ", pred_acc - random_acc)
        return pred_acc / random_acc, pred_acc - random_acc


def main():
    analyzer = Analyzer()
    analyzer.plot_new_occurrences(topx=10, lowx=10)
    # analyzer.plot_occurrences(topx=5, lowx=5)
    # analyzer.plot_occurrences(topx=10, lowx=10)


if __name__ == "__main__":
    main()
