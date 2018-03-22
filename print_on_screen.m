function print_on_screen(acc, num_task, annotations, id)

method = {
  'NA     '
  'LDADA  '
  'ORACLE '
};
        
if nargin > 2
  fprintf('--------------------------------------\n')
  fprintf([annotations.prm.sourceName '-->' annotations.prm.targetName '\n'])
  for i = 1:length(method)
    fprintf( ...
      ['acccuracy - ' method{i} ' = %3.1f(%3.1f)\n'], ...
      acc{i,id}{1}, ...
      acc{i,id}{2} ...
    )
  end
  fprintf('--------------------------------------\n')
else
  avg_acc = 0;
  for i = 1:length(method)
    for j = 1:num_task
      avg_acc = avg_acc + acc{i,j}{1};
    end
    acc{i,num_task+1} = avg_acc / num_task;
    avg_acc = 0;
  end
  fprintf('--------------------------------------\n')
  for i = 1:length(method)
    fprintf( ...
      ['mean average accuracy - ' method{i} ' = %3.1f\n'], ...
      acc{i,num_task+1} ...
    )
  end
  fprintf('--------------------------------------\n')
end