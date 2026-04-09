clc; clear; close all;

% Load dataset
data = readtable('../data/employee_attrition.csv');

% Convert Attrition to binary if needed (Yes/No → 1/0)
if iscell(data.Attrition)
    y = strcmp(data.Attrition, 'Yes');
else
    y = strcmp(data.Attrition, 'Yes'); % Convert Yes/No → 1/0
end

% Select predictors
X = [data.Age, data.MonthlyIncome, data.YearsAtCompany, data.JobSatisfaction];

% Add intercept
X = [ones(size(X,1),1) X];

%% ===== Built-in Logistic Regression =====
[b, dev, stats] = glmfit(X(:,2:end), y, 'binomial');

disp('--- GLM Coefficients ---');
disp(b);

% Predictions
p = glmval(b, X(:,2:end), 'logit');
y_pred = p >= 0.5;

%% ===== Evaluation =====
[acc, sens, spec, confMat] = evaluation(y, y_pred);

disp('Confusion Matrix:');
disp(confMat);

fprintf('Accuracy: %.4f\n', acc);
fprintf('Sensitivity: %.4f\n', sens);
fprintf('Specificity: %.4f\n', spec);

%% ===== ROC Curve =====
[Xroc, Yroc, ~, AUC] = perfcurve(y, p, 1);

figure;
plot(Xroc, Yroc, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);
grid on;

% Save plot
saveas(gcf, '../results/roc_curve.png');

%% ===== Save Results =====
writematrix(confMat, '../results/confusion_matrix.txt');

fileID = fopen('../results/model_summary.txt','w');
fprintf(fileID, 'Coefficients:\n');
fprintf(fileID, '%f\n', b);
fprintf(fileID, '\nAccuracy: %.4f\n', acc);
fprintf(fileID, 'Sensitivity: %.4f\n', sens);
fprintf(fileID, 'Specificity: %.4f\n', spec);
fprintf(fileID, 'AUC: %.4f\n', AUC);
fclose(fileID);
