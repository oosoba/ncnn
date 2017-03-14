%function err_rates_nem = ParSketch(log_file)

n = 20;
c_nom = 1000:-50: 500 ;

clean_err = zeros(n,length(c_nom));
err_rates_nem = zeros(n,length(c_nom));
parfor k = 1:n
    clean_err(k,:) = ...
        CNNSketchLogged(c_nom, sprintf(['log/clean-ref-nem-1e3-down.txt']));
    err_rates_nem(k, :) = ...
        CNEMSketchLoggedV2(c_nom, sprintf(['log/parCleanSketchNEM-', num2str(k), '.txt']));
end

c = 1:length(c_nom);
figure;
plot(c, err_rates_nem', '.');
hold on;
plot(c, mean(err_rates_nem), 'rd-');
plot(c, trimmean(err_rates_nem, 30), 'g+');
plot(c, trimmean(err_rates_nem, 30), 'gd-');
plot(c, mean(clean_err), 'b-');
hold off;
fid = fopen('log/sketch_rates_nem-down-1e3.csv', 'a');
fwrite(fid,err_rates_nem);
fclose(fid);

%return;
