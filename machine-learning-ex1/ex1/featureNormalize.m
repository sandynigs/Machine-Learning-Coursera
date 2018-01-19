function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
features = size(X,2);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%Now the code will loop through any number of features and will 
%update matrix for all of them.

for i = 1:features
	mu(:,i) = mean(X(:,i));
	X_norm(:,i) = X(:,1) - mean(X(:,i));
	sigma(:,i) = std(X(:,i));
	X_norm(:,i) = X_norm(:,i)./std(X(:,i));
end


%we only divide standard 
%deviation after subtracting mean, 
%i.e why we have divide here from X_norm









% ============================================================

end
