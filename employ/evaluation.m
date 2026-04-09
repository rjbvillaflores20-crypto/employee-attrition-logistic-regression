function [accuracy, sensitivity, specificity, confMat] = evaluation(y, y_pred)

TP = sum((y == 1) & (y_pred == 1));
TN = sum((y == 0) & (y_pred == 0));
FP = sum((y == 0) & (y_pred == 1));
FN = sum((y == 1) & (y_pred == 0));

confMat = [TP FN; FP TN];

accuracy = (TP + TN) / (TP + TN + FP + FN);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

end
