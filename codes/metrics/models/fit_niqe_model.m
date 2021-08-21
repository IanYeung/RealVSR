function fit_niqe_model(realvsr_gt_dir)
    filepaths = dir(fullfile(realvsr_gt_dir, '*', '*.png'));
    img_cell_1 = {};
    for i = 1 : length(filepaths)
        [~,imname,ext] = fileparts(filepaths(i).name);
        folder_path = filepaths(i).folder;
        img_cell_1{i} = fullfile(folder_path, [imname, ext]);
    end
    img_cell = [img_cell_1];
    imds = imageDatastore(img_cell);
    niqe_model = fitniqe(imds);  % default block size is 96x96
    save('./niqe_model_realvsr_all.mat', 'niqe_model');
end