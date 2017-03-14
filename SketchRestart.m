%function err_rates_nem = ParSketch(log_file)

n = 5;
c = [850, 900, 950, 1000];
cb = [4000, 4500, 5000];


bench_err = zeros(n,length(c));
big_err = zeros(n,length(cb));
parfor k = 1:n
    bench_err(k,:) = ...
        CNNSketchLogged(c, sprintf(['log-v1/sketch-1e3-bench.txt']));
end

parfor k = 1:n
    big_err(k,:) = ...
        CNNSketchLogged(cb, sprintf(['log-v1/sketch-5e3-try.txt']));
end

figure;
hold on;
plot(cb, mean(big_err), 'rd-');
plot(cb, trimmean(big_err, 30), 'gd-');
plot(c, mean(bench_err), 'b-');
hold off;
% fid = fopen('log-v1/sketch_rates_nem-vFill.csv', 'a');
% fwrite(fid,err_rates_nem);
% fclose(fid);

%return;
