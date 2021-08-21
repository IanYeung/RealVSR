function generate_LR_BI_Vimeo90K()
    %% matlab code to genetate bicubic-downsampled for Vimeo90K dataset
    up_scale = 2;
    mod_scale = 2;
    idx = 0;
    filepaths = dir('/home/xiyang/Datasets/Vimeo90k/vimeo_septuplet/sequences/*/*/*.png');
    for i = 1 : length(filepaths)
        [~,imname,ext] = fileparts(filepaths(i).name);
        folder_path = filepaths(i).folder;
        save_LR_folder = strrep(folder_path,'vimeo_septuplet','vimeo_septuplet_BIx2_same');
        if ~exist(save_LR_folder, 'dir')
            mkdir(save_LR_folder);
        end
        if isempty(imname)
            disp('Ignore . folder.');
        elseif strcmp(imname, '.')
            disp('Ignore .. folder.');
        else
            idx = idx + 1;
            str_rlt = sprintf('%d\t%s.\n', idx, imname);
            fprintf(str_rlt);
            % read image
            img = imread(fullfile(folder_path, [imname, ext]));
            img = im2double(img);
            % modcrop
            img = modcrop(img, mod_scale);
            % LR
            im_LR = imresize(imresize(img, 1/up_scale, 'bicubic'), up_scale, 'bicubic');
            if exist('save_LR_folder', 'var')
                imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
            end
        end
    end
end

function img = modcrop(img, modulo)
    %% modcrop
    if size(img,3) == 1
        sz = size(img);
        sz = sz - mod(sz, modulo);
        img = img(1:sz(1), 1:sz(2));
    else
        tmpsz = size(img);
        sz = tmpsz(1:2);
        sz = sz - mod(sz, modulo);
        img = img(1:sz(1), 1:sz(2),:);
    end
end