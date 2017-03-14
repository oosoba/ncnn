%function err_rates_nem = ParSketch(log_file)

n = 15;
c = 5e2:1e2:1.1e3;

clean_err = zeros(n,length(c));
err_rates_nem = zeros(n,length(c));
parfor k = 1:n
    clean_err(k,:) = ...
        CNNSketchLogged(c, sprintf(['log/clean-ref-nem-1e3-v2.txt']));
    err_rates_nem(k, :) = ...
        CNEMSketchLoggedV2(c, sprintf(['log/parCleanSketchNEM-', num2str(k), '.txt']));
end

figure;
plot(c, err_rates_nem', '.');
hold on;
plot(c, mean(err_rates_nem), 'rd-');
plot(c, trimmean(err_rates_nem, 30), 'g+');
plot(c, trimmean(err_rates_nem, 30), 'gd-');
plot(c, mean(clean_err), 'b-');
hold off;
fid = fopen('log/sketch_rates_nem-v2.csv', 'a');
fwrite(fid,err_rates_nem);
fclose(fid);

%return;
