W = -matrix_to_plot;
for i=1:21
    W(i,i) = 0;
end
G = gsp_graph(W);
G = gsp_update_coordinates(G,rand(21,2));
gsp_plot_graph(G)
%gsp_plot_signal(G,signal_to_plot)
