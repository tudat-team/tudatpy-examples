"""Compare stored active precision with marginal covariance of retained observations."""
from pathlib import Path
import json,sys,numpy as np
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve

def inverse(matrix):
    scales=np.sqrt(np.diag(matrix));balanced=matrix/np.outer(scales,scales)
    return cho_solve(cho_factor(balanced),np.eye(len(matrix)))/np.outer(scales,scales)

def check(directory):
    f=np.load(directory/'fit.npz');o=np.load(directory/'observations.npz');allw=sparse.load_npz(directory/'observation_weights.npz');w=sparse.load_npz(directory/'fit_weights.npz').tolil();best=int(f['best_iteration']);rows=np.flatnonzero(f['active_flags_per_iteration'][:,best]);setids=o['all_set_ids'];h=f['normalized_design_matrix'];fac=f['normalization'].reshape(-1);q=f['best_parameters'].reshape(-1);r=f['best_residuals'].reshape(-1);differences=[]
    for sid in np.unique(setids[rows]):
        whole=np.flatnonzero(setids==sid);active_positions=np.flatnonzero(setids[rows]==sid)
        if len(whole)==len(active_positions):continue
        sparse_block=allw[whole][:,whole]
        if (sparse_block-sparse.diags(sparse_block.diagonal())).nnz==0:continue
        block=sparse_block.toarray()
        within=np.searchsorted(whole,rows[active_positions]);cov=inverse(block);expected=inverse(cov[np.ix_(within,within)]);current=w[active_positions][:,active_positions].toarray()
        difference=float(np.linalg.norm(current-expected)/np.linalg.norm(expected));differences.append(dict(set_id=int(sid),total_scalars=len(whole),kept_scalars=len(within),relative_precision_difference=difference))
        w[np.ix_(active_positions,active_positions)]=expected
    w=w.tocsc();p=f['prior_inverse_covariance']/np.outer(fac,fac);n=h.T@(w@h)+p;c=inverse(n);rhs=h.T@(w@r)+p@((f['parameter_history'][:,0]-q)*fac);delta=c@rhs;i=list(f['parameter_names']).index('beta')
    result=dict(directory=str(directory),correlated_partial_sets=differences,original_beta=float(q[i]),original_sigma_beta=float(f['formal_errors'][i]),corrected_linearized_beta=float(q[i]+delta[i]/fac[i]),corrected_sigma_beta=float(np.sqrt(c[i,i])/abs(fac[i])),method='Marginalize each full observation covariance before selecting retained rows; then recompute linearized fit with fixed rejection mask.')
    (directory/'rejected_covariance_check.json').write_text(json.dumps(result,indent=2)+'\n');sparse.save_npz(directory/'marginal_fit_weights.npz',w);print(json.dumps(result,indent=2))
if __name__=='__main__':check(Path(sys.argv[1]))
