function extract_features(image_path, output_excel)
    % Read the image
    image = imread(image_path);
    image = im2gray(image);  % Convert to grayscale if needed
    image = imresize(image, [256, 256]);  % Resize to 256x256 if necessary

    block_size = 8;
    num_blocks = (256 / block_size) ^ 2; % 1024 blocks
    feature_matrix = zeros(num_blocks, 5); % Store features

    block_idx = 1;

    for i = 1:block_size:256
        for j = 1:block_size:256
            % Extract 8x8 block
            block = image(i:i+block_size-1, j:j+block_size-1);

            % Compute GLCM
             GLCM = graycomatrix(block, 'Offset', [0 1]); % Horizontal adjacency

            % Extract features from GLCM
            stats = graycoprops(GLCM, {'contrast', 'energy', 'correlation', 'homogeneity'});

            % Compute entropy
            entropy_value = entropy(block);

            % Store results
            feature_matrix(block_idx, :) = [entropy_value, stats.Energy, stats.Correlation, stats.Contrast, stats.Homogeneity];

            block_idx = block_idx + 1;
        end
    end

    % Replace NaN values with 0
    feature_matrix(isnan(feature_matrix)) = 0;

    % Define headers
    headers = {'Entropy', 'Energy', 'Correlation', 'Contrast', 'Homogeneity'};

    % Save to Excel
    writecell([headers; num2cell(feature_matrix)], output_excel);

    disp([' Feature extraction complete. Data saved to ', output_excel]);
end
