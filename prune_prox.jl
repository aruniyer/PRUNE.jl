using ArgParse, CSV, Knet, Statistics

# Defining the PRUNE model
struct Linear; w; b; end;
Linear(inputsize::Int, outputsize::Int) = 
    Linear(param(outputsize, inputsize), param0(outputsize));
(model::Linear)(x) = 
    model.w * x .+ model.b;

struct Proximity; linear1; linear2; end;
(model::Proximity)(x) = 
    relu.(model.linear2(elu.(model.linear1(x))));

struct Embedding; E; end;
Embedding(embedding_size::Int, instances::Int) = 
    Embedding(param(embedding_size, instances));
(model::Embedding)(x) = model.E[:, x];

struct PRUNE; e; p; W; end;
PRUNE(embedding_size::Int, instances::Int, hidden_size::Int) = 
    PRUNE(Embedding(embedding_size, instances), 
        Proximity(
            Linear(embedding_size, hidden_size), 
            Linear(hidden_size, hidden_size)), 
        param(hidden_size, hidden_size));
(model::PRUNE)(x) = 
    model.p(model.e(x[1]))'*(relu.(model.W))*model.p(model.e(x[2]));

ProximityLoss(model, s, t, pmi) = 
    mean((model.(tuple.(s, t)) - pmi).^2)

function calc_pmi(source, target, nodeCount, in_degrees, out_degrees, alpha)
    n = length(source);
    PMI_values = zeros(Float32, n);
    for i = 1:n
        s, t = source[i], target[i];
        pmi = n / (alpha * out_degrees[s] * in_degrees[t]);
        PMI_values[i] = log(pmi);
    end

    PMI_values[PMI_values .< 0] .= 0;

    return PMI_values;
end

function train(model, source, target, pmi, learning_rate, epochs)
    for epoch = 1:epochs
        println("epoch = ", epoch, 
            ", loss = ", ProximityLoss(model, source, target, pmi));
        loss = @diff ProximityLoss(model, source, target, pmi);
        for param in (model.W, 
                    model.p.linear2.w, model.p.linear2.b, 
                    model.p.linear1.w, model.p.linear1.b, 
                    model.e.E)
            ∇param = grad(loss, param);
            axpy!(-1.0 * learning_rate, ∇param, value(param));
        end
    end
end

# Executing the PRUNE model on the data
function parse_command_line()
    s = ArgParseSettings();
    @add_arg_table s begin
        "--ig"
            help = "input graph file"
            arg_type = String
            required = true
        "--d"
            help = "size of the embedding"
            arg_type = Int64
            default = 100
        "--h"
            help = "size of the hidden representation"
            arg_type = Int64
            default = 64
        "--a"
            help = "alpha parameter of PRUNE (num negative samples)"
            arg_type = Float64
            default = 5.0
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 0.01
        "--ep"
            help = "number of epochs"
            arg_type = Int64
            default = 50
    end

    return parse_args(s);
end

function main()
    o = parse_command_line();
    println("opts = ", [(k,v) for (k,v) in o]...)
    graph = CSV.read(o["ig"], header=0, delim=' ');
    source = convert(Array{Int, 1}, graph[1]);
    target = convert(Array{Int, 1}, graph[2]);
    n = max(maximum(source), maximum(target));
    embedding_size = o["d"];
    hidden_size = o["h"];
    M = length(source);
    out_degrees = zeros(Float32, n);
    in_degrees = zeros(Float32, n);
    for i = 1:M
        node_i, node_j = source[i], target[i];
        out_degrees[node_i] += 1;
        in_degrees[node_j] += 1;
    end
    pmi = calc_pmi(source, target, n, in_degrees, out_degrees, o["a"]);
    model = PRUNE(embedding_size, n, hidden_size);
    train(model, source, target, pmi, o["lr"], o["ep"]);
end

main()