"""
    interpolate_BR_to_P2!(u_P2::FEVectorBlock, u_BR::FEVectorBlock)

Interpolate a vector-valued Bernardi--Raugel (BR/RT0) finite element solution
onto a refined barycentric P2 grid. The coarse-grid nodal DOFs are copied
verbatim; new mid-edge and interior DOFs are obtained by evaluating the BR
basis at reference coordinates and mapping to the refined grid.
"""
function interpolate_BR_to_P2!(u_P2::FEVectorBlock, u_BR::FEVectorBlock)
    FES_BR = u_BR.FES
    FES_P2 = u_P2.FES
    xgrid_bary = u_P2.FES.xgrid
    xgrid = u_BR.FES.xgrid
    cellparents = xgrid_bary[CellParents]
    cellnodes_bary = xgrid_bary[CellNodes]

    dofs_BR = view(u_BR)
    dofs_P2 = view(u_P2)
    facedofs_P2 = FES_P2[FaceDofs] # NNNNNNFFF

    # node dofs of coarse grid remain unchanged
    nnodes = num_nodes(xgrid)
    nnodes_bary = num_nodes(xgrid_bary)
    nfaces_bary = num_sources(facedofs_P2)
    for n = 1 : nnodes
        dofs_P2[n] = dofs_BR[n]
        dofs_P2[n+(nnodes_bary+nfaces_bary)] = dofs_BR[n + nnodes]
    end

    ## define PointEvaluator for uBR
    PE = PointEvaluator([id(1)], [u_BR])
    evalBR = zeros(Float64, 2)
    xref_center = [1/3, 1/3]
    xref_outer = [[1/2, 0], [1/2, 1/2], [0, 1/2]]
    xref_inner = [[1/6, 1/6], [4/6, 1/6], [1/6, 4/6]]
    cell::Int = 0

    ncells_bary = num_cells(xgrid_bary)
    cellfaces_bary = xgrid_bary[CellFaces]
    node_offset = (nnodes_bary+nfaces_bary)
    face_offset = node_offset + nnodes_bary
    for cell_bary = 1 : ncells_bary
        cell = cellparents[cell_bary]

        # determine on which part of the coarse triangle we are
        child_type = mod(cell_bary - 1, 3) + 1 

        if child_type == 1
            # set new vertex in center (xref = [1/3, 1/3, 1/3])
            new_node = cellnodes_bary[3, cell_bary]
            evaluate_bary!(evalBR, PE, xref_center, cell)
            dofs_P2[new_node] = evalBR[1]
            dofs_P2[node_offset + new_node] = evalBR[2]
        end

        # set dof on outer face
        face_outer = cellfaces_bary[1, cell_bary]
        evaluate_bary!(evalBR, PE, xref_outer[child_type], cell)
        dofs_P2[nnodes_bary + face_outer] = evalBR[1]
        dofs_P2[face_offset + face_outer] = evalBR[2]
    
        # set dof on inner face
        face_inner = cellfaces_bary[3, cell_bary]
        evaluate_bary!(evalBR, PE, xref_inner[child_type], cell)
        dofs_P2[nnodes_bary + face_inner] = evalBR[1]
        dofs_P2[face_offset + face_inner] = evalBR[2]
    end
end
export interpolate_BR_to_P2!
