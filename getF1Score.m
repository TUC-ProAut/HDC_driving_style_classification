function f1 = getF1Score(y_gt, y_pred)
% get the weighted F1 score

CM = confusionmat(double(y_gt),double(y_pred));

num_classes = size(CM,1);
[TP,TN,FP,FN,f1] = deal(zeros(num_classes,1));
for c = 1:num_classes
   TP(c) = CM(c,c);
   tempMat = CM;
   tempMat(:,c) = []; % remove column
   tempMat(c,:) = []; % remove row
   TN(c) = sum(sum(tempMat));
   FP(c) = sum(CM(:,c))-TP(c);
   FN(c) = sum(CM(c,:))-TP(c);
end
for c = 1:num_classes
    f1(c) = 2*TP(c)/(2*TP(c) + FP(c) + FN(c));
end

h=hist(double(y_gt),0:num_classes-1)';
f1 = sum(stats.Fscore.*(h/sum(h)));

end 