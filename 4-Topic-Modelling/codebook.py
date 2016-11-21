def create_codebook(V, distf, epsilon, alpha):
	C = []
	F = []
	C.append(V[0])
	F.append(1)

	for vi in V:
		
		alldist = [distf(vi, cj) for cj in C]

		if(min(alldist) > epsilon):
			C.append(vi)
			F.append(1)
		else:
			sumdist = sum(1./alldist)
			wi = (1./alldist)*(1/sumdist)
			C = (F.*C + wi*vi)./(F + wi)
			F = F + wi

	for ci in C:
		for cj in C:
			if(distf(ci, cj) < alpha * epsilon and F[j] < 0.1*N/M):
				merge






