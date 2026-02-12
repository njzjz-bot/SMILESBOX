from ase import Atoms
from ase.data import atomic_numbers
from openbabel import pybel
import numpy as np 

class SMILESbox:
    
    def __init__(self,
            smiles:str|list[str]|tuple[str,...],
            box_size:float=20.0,
            min_distance:float=2.0,
            random_rotate:bool=True,
            axis:str='x',
            margin:float=1.5):
        
        self.molecule = None
        if isinstance(smiles,(list,tuple)):
            if len(smiles) < 1:
                raise ValueError("When passing a list/tuple, provide at least one SMILES string.")
            if not all(isinstance(s,str) for s in smiles):
                raise TypeError("All entries in smiles list/tuple must be strings.")
            self.smiles = list(smiles)
            self.atoms = self.add_multiple_to_box(
                smiles_list=self.smiles,
                box_size=box_size,
                min_distance=min_distance,
                random_rotate=random_rotate,
                axis=axis,
                margin=margin,
            )
        else:
            if not isinstance(smiles,str):
                raise TypeError("smiles must be a SMILES string, or a list/tuple of SMILES strings.")
            self.smiles = smiles 
            self.atoms = self._smiles_to_atoms()


    def _smiles_to_atoms(
            self,
            )->Atoms:
        
        molecule = pybel.readstring("smi",self.smiles)
        molecule.make3D()
        symbols = []
        positions = []
        for i in molecule.atoms:
            coords = list(i.coords)
            atom = [k for k,v in atomic_numbers.items() if v == i.atomicnum]
            symbols.append(atom[0])
            positions.append(coords)

        atoms = Atoms(symbols=symbols,positions=positions)
        self.molecule = molecule
        return(atoms)


    def _smiles_to_atoms_from_string(self,smiles:str)->Atoms:
        molecule = pybel.readstring("smi",smiles)
        molecule.make3D()
        symbols = []
        positions = []
        for i in molecule.atoms:
            coords = list(i.coords)
            atom = [k for k,v in atomic_numbers.items() if v == i.atomicnum]
            symbols.append(atom[0])
            positions.append(coords)
        return(Atoms(symbols=symbols,positions=positions))
    

    def generate_random_points(self,N, L, min_dist, max_attempts=100000):
        """
        Generate N random 3D points inside a cube of size L, with a minimum spacing.    

        Parameters:
        - N: Number of points.
        - L: Length of the cube along each axis (cube from 0 to L in x, y, z).
        - min_dist: Minimum allowed distance between any two points.
        - max_attempts: Max number of iterations before giving up.    

        Returns:
        - points: (N, 3) array of valid 3D coordinates.
        """
        points = []
        attempts = 0    

        while len(points) < N and attempts < max_attempts:
            candidate = np.random.uniform(0, L, size=3)
            if all(np.linalg.norm(candidate - np.array(p)) >= min_dist for p in points):
                points.append(candidate)
            attempts += 1    

        if len(points) < N:
            raise RuntimeError(f"Could not place all points within {max_attempts}     attempts. "
                               f"Try reducing N or min_dist.")
        return np.array(points)
    
    def multiple_in_a_box(self,num_points,box_size,min_distance=0.5,random_rotate=True):
        final_cell = Atoms(cell=[box_size,box_size,box_size])
        try:
            points = self.generate_random_points(N=num_points,L=box_size,min_dist=min_distance)
        except RuntimeError as e:
            raise RuntimeError(f'{e} or try a larger box...{box_size} may be too small')
            
        for point in points:
            _atoms = self._smiles_to_atoms()
            _atoms = self.add_box(_atoms,[box_size,box_size,box_size])
            if random_rotate:
                _atoms = self.rotate(_atoms,np.random.choice(['x','y','z']),np.random.randint(360))
            _atoms = self.translate(_atoms,point)
            final_cell+= _atoms
        return(final_cell)


    def add_multiple_to_box(self,
            smiles_list:list[str]|tuple[str,...]=None,
            box_size:float=20.0,
            min_distance:float=2.0,
            random_rotate:bool=True,
            axis:str|None=None,
            margin:float=1.5,
            max_attempts:int=100000,
            smiles_1:str=None,
            smiles_2:str=None,
            )->Atoms:
        """
        Add one or more molecules from SMILES to one periodic box with reasonable separation.
        """
        if smiles_list is None:
            if smiles_1 is not None and smiles_2 is not None:
                smiles_list = [smiles_1,smiles_2]
            else:
                raise ValueError("Provide smiles_list, or provide both smiles_1 and smiles_2.")

        if not isinstance(smiles_list,(list,tuple)) or len(smiles_list) < 1:
            raise ValueError("smiles_list must be a list/tuple containing at least one SMILES string.")
        if not all(isinstance(s,str) for s in smiles_list):
            raise TypeError("All entries in smiles_list must be strings.")

        axis_map = {
            'x': np.array([1.0,0.0,0.0]),
            'y': np.array([0.0,1.0,0.0]),
            'z': np.array([0.0,0.0,1.0]),
        }
        if axis is not None and axis not in axis_map:
            raise ValueError("axis must be one of: 'x', 'y', 'z', or None")

        molecules = [self._smiles_to_atoms_from_string(s) for s in smiles_list]
        for mol in molecules:
            if random_rotate:
                for rot_axis in ['x','y','z']:
                    self.rotate(mol,rot_axis,np.random.randint(360))

        radii = []
        for mol in molecules:
            com = mol.get_center_of_mass()
            radii.append(np.max(np.linalg.norm(mol.get_positions() - com,axis=1)))
        radii = np.array(radii)

        final_cell = Atoms(cell=[box_size,box_size,box_size],pbc=True)
        center = np.array([box_size / 2.0,box_size / 2.0,box_size / 2.0])

        # Optional deterministic placement for exactly two molecules.
        centers = []
        if len(molecules) == 2 and axis in axis_map:
            separation = max(min_distance,radii[0] + radii[1] + margin)
            if separation >= box_size:
                raise RuntimeError(
                    f"Required separation ({separation:.2f} A) is too large for box_size={box_size}. "
                    "Increase box_size or lower min_distance/margin."
                )
            shift = axis_map[axis] * (separation / 2.0)
            centers = [center - shift,center + shift]
        else:
            for i,_ in enumerate(molecules):
                radius = radii[i]
                low = radius + margin
                high = box_size - (radius + margin)
                if low >= high:
                    raise RuntimeError(
                        f"box_size={box_size} is too small for molecule index {i}. "
                        "Increase box_size or reduce margin."
                    )

                placed = False
                for _ in range(max_attempts):
                    candidate = np.random.uniform(low,high,size=3)
                    ok = True
                    for j,other_center in enumerate(centers):
                        required = max(min_distance,radius + radii[j] + margin)
                        if np.linalg.norm(candidate - other_center) < required:
                            ok = False
                            break
                    if ok:
                        centers.append(candidate)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError(
                        f"Could not place all molecules without overlap after {max_attempts} attempts. "
                        "Try a larger box or reduce min_distance/margin."
                    )

        for mol,target in zip(molecules,centers):
            mol = self.add_box(mol,[box_size,box_size,box_size])
            mol = self.translate(mol,target)
            final_cell += mol

        return(final_cell)
    
    def add_box(
            self,
            atoms:Atoms=None,
            dimensions:np.ndarray=None,
            )->None:
        """
        dimensions = np.ndarray([[10,0,0],
        [0,10,0],
        [0,0,10]])

        i.e. 
        import numpy as np 
        dimensions = np.eye(3) * 10 
        """
        if not atoms:
            atoms = self.atoms    
        atoms.set_cell(dimensions)
        atoms.set_pbc(True)
        atoms.center()
        self.atoms = atoms
        return(atoms)


    
    def translate(self,atoms,new_coords):
        com = atoms.get_center_of_mass()
        vector = new_coords - com 
        atoms.translate(vector)
        return(atoms)

    
    def rotate(self,atoms,vector='x',angle=90):
        atoms.rotate(a=angle,v=vector)
        return(atoms)
