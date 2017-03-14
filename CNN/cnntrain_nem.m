% Noisy version of CNN training

function net = cnntrain_nem(net, x, y, opts)
	noise_var = opts.noise_var;
	anneal_fact = opts.anneal_fact;
	noise_type = opts.noise_type;
	noise_pdf = opts.noise_pdf;
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
	fp = fopen(opts.logfile,'w');
    for i = 1 : opts.numepochs
		net.batch_loss = zeros(1,numbatches);
		net.batch_error = zeros(1,numbatches);
        tic;
%        kk = randperm(m);
		kk = 1:m;
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            [net, noise] = cnnff_nem(net, batch_x, batch_y, noise_var/(i^anneal_fact), noise_type, noise_pdf);
            net = cnnbp(net, batch_y+noise);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;

			% Compute the loss function with clean labels
			% since I passed the noisy labels to cnnbp which computes
			% the loss function net.L
			net.e = batch_y - net.o;
		    switch net.output
				case 'sigm'
					%  Squared-error loss function
					net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
			    case 'softmax'
					% Cross-entropy loss function
					net.L = -sum(sum(batch_y .* log(net.o))) / size(net.e, 2);
			end
			net.batch_loss(l) = net.L;
			[net.batch_error(l),~] = cnntest(net, batch_x, batch_y);
        end
        dur = toc;
        disp(['epoch:' num2str(i) '/' num2str(opts.numepochs) ', mean batch loss:' num2str(mean(net.batch_loss)) ', mean batch error:' num2str(mean(net.batch_error)) ', time taken:' num2str(dur)]);
		fprintf(fp,'%d\t%f\t%f\t%f\n',i,mean(net.batch_loss),mean(net.batch_error),dur);
    end
	fclose(fp);
end
