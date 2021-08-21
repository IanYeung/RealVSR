function evaluate_niqe_brisque(root, expn, if_niqe, if_brisque, result_path, niqe_model_path)
    
    fileID = fopen(result_path,'a');
    
    root_dir = strcat(root, expn);

    seq_struct = dir(fullfile(root_dir, '*'));
    seq_struct = seq_struct([seq_struct.isdir]);

    if niqe_model_path ~= -1
        load(niqe_model_path, 'niqe_model');
        fprintf('load custem model\n');
    else
        niqe_model = niqeModel;
        fprintf('load default model\n');
    end

    % filter away '.' and '..'
    seq_cell = regexpi({seq_struct.name}, '[0-9]{3}', 'match', 'once');
    seq_cell = seq_cell(~cellfun('isempty', seq_cell));
    
    if if_niqe
        niqe_results = cell(1, length(seq_cell));
    end
    if if_brisque
        brisque_results = cell(1, length(seq_cell));
    end
    % loop for sequence
    for i = 1:length(seq_cell)

        frm_struct = dir(fullfile(root_dir, seq_cell{i}, '*.png')); 
        frm_cell = {frm_struct.name};

        niqe_folder_sum = 0;
        brisque_folder_sum = 0;

        % loop for frame
        for j = 1:length(frm_cell)
           frm_path = fullfile(root_dir, seq_cell{i}, frm_cell{j});
           img = imread(frm_path);
           if if_niqe
               niqe_score = niqe(img, niqe_model);
               niqe_folder_sum = niqe_folder_sum + niqe_score;
           end
           if if_brisque
               brisque_score = brisque(img);
               brisque_folder_sum = brisque_folder_sum + brisque_score;
           end
        end
        
        if if_niqe
            fprintf('%s NIQE: %.4f\n', seq_cell{i}, niqe_folder_sum / length(frm_cell));
            fprintf(fileID, '%s NIQE: %.4f\n', seq_cell{i}, niqe_folder_sum / length(frm_cell));
            niqe_results{i} = niqe_folder_sum / length(frm_cell);
        end
        if if_brisque
            fprintf('%s BRISQUE: %.4f\n', seq_cell{i}, brisque_folder_sum / length(frm_cell));
            fprintf(fileID, '%s BRISQUE: %.4f\n', seq_cell{i}, brisque_folder_sum / length(frm_cell));
            brisque_results{i} = brisque_folder_sum / length(frm_cell);
        end
    end
    
    if if_niqe
        niqe_mean = mean(cell2mat(niqe_results));
        fprintf(fileID, '%s, NIQE: %.4f\n', expn, niqe_mean);
    end
    if if_brisque
        brisque_mean = mean(cell2mat(brisque_results));
        fprintf(fileID, '%s, BRISQUE: %.4f\n', expn, brisque_mean);
    end
    fclose(fileID);
end

