% Computes the mean difference between two log files generated
% by NN or BAM training over a specified number of iterations.

function [meandiff] = compute_mean_diff(logfile1,logfile2,maxiter)
	log1 = textread(logfile1);
	log2 = textread(logfile2);
	if maxiter > size(log1,1) || maxiter > size(log2,1)
		fprintf('Error: maxiter exceeds length of atleast one log file');
		meandiff = -1;
	else
		meandiff = mean(log1(1:maxiter,2)-log2(1:maxiter,2));
	end
end
