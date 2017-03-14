%function err_rates_bnem = ParSketch(log_file)
n = 25;
c = 7e2:50:1.1e3;

clean_err = ...
    CNNSketchLogged(1e3*ones(n,1), sprintf(['log/clean-ref-bnem-1e3-v1.txt']));

err_rates_bnem = zeros(n,length(c));
parfor k = 1:n
    err_rates_bnem(k, :) = ...
        CNEMSketchLogged(c, sprintf(['log/parBlindNEMSketch-', num2str(k), '.txt']));
end

figure;
plot(c, err_rates_bnem', '.');
hold on;
plot(c, mean(err_rates_bnem), 'rd-');
plot(c, trimmean(err_rates_bnem, 30), 'g+');
plot(c, trimmean(err_rates_bnem, 30), 'gd-');
plot(c, mean(clean_err)*ones(1,length(c)), 'b-');
hold off;
fid = fopen('log/sketch_rates_blind_nem-v1.csv', 'a');
fwrite(fid,err_rates_bnem);
fclose(fid);

%return;
