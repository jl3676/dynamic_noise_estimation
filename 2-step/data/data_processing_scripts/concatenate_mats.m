mats = dir('../online_mat_files/*.mat')';

ages = readtable('../mbmf_ages.csv');


%%
data = struct('data',{},'age',{},'group',{});
i = 1;
for file=mats
    data(i).data = load([mats(i).folder '/' mats(i).name]);
    this_id = mats(i).name(1:end-11);
    data(i).age = ages.age(find(strcmp(ages.subject_id,this_id)));
    if data(i).age < 13
        data(i).group = 1;
    elseif data(i).age < 18
        data(i).group = 2;
    else
        data(i).group = 3;
    end
    i = i + 1;
end

%%
save('../Alldata.mat', "data")