
def loadAnnotations(filename = None, text = None):
    """
        This function loads annotation from a *.txt file ('filename'). It returns a list of 'FrameAnnotations'
        objects, which contain the following information:
          1. a <Nx2> array of <x,y> coordinates for each junction
          2. a <Mx2> array of <id1,id2>, indicating the junction IDs that are connected by a silk line.
          3. a <Mx4> array of <x1,y1,x2,y2>, representing all the lines in the file.
             (NOTE: this is redundant but provided for convenience.)
          4. a list of lists: for every junction, it returns a list of line coordinates (x1, y1, x2, y2) connected
             to that junction. This facilitates computing angles, etc.
          5. a list of lists of angles: one list of angles per junction
        The variables that can be accessed for each FrameAnnotation are:
            'frame', 'junctions', 'lines', 'lengths', 'lines_coords', 'lines_by_junction'
    """
    import os, numpy as np
    from collections import namedtuple, Counter
    def vecAngle(v1_, v2_):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        v1 = v1_ / np.linalg.norm(v1_)
        v2 = v2_ / np.linalg.norm(v2_)
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)
    FrameAnnotations = namedtuple('FrameAnnotations', [
        'frame', 'junctions', 'lines', 'lines_coords', 'lines_by_junction', 'angles_by_junction', 'lengths'])
    if filename is not None and not os.path.exists(filename):
        raise Exception('File not found!')
    else:
        if filename is not None:
            text = open(filename, 'r').read()
        # Split text into frames
        if 'frame=' not in text:
            text = 'frame=-1\n' + text
        textFrames = [y for y in [x.strip() for x in text.split('frame=')] if len(y) > 0]
        # Parse each annotation
        annotations = []
        for text in textFrames:
            fid = int(text[:text.find('\n')])
            text = text[text.find('\n') + 1:] + '\n'
            rois, kind, pts = [], None, None
            for text_line in text.split('\n'):
                if kind is None:
                    kind, pts = text_line, []
                elif text_line == '':
                    rois.append((kind, pts));
                    kind, pts = None, None
                else:
                    pts.append(tuple(int(float(i)) for i in text_line.split()))
            # Obtain all lines in this frame
            lineCoords = np.array([x[1][0] + x[1][1] for x in rois if x[0] == 'line'], dtype=np.int64)
            # Obtain unique points for this line
            pts = np.unique(np.vstack((lineCoords[:, 0:2], lineCoords[:, 2:4])), axis=0)
            # Obtain lines, but specified using <ID1, ID2> rather than <x1, y1, x2, y2>
            lines = np.array([[np.argwhere((pts[:, 0] == x[j]) & (
                    pts[:, 1] == x[j + 1]))[0, 0] for j in [0, 2]] for x in lineCoords], dtype=np.int64)
            # Compute line lengths
            lineLengths = np.linalg.norm(lineCoords[:,0:2] - lineCoords[:,2:4], axis=1)
            # For every junction, obtain the lines that are connected to it
            lineCoordsByJunction = [np.vstack([np.hstack((pts[x[0]], pts[x[1]])) for x
                                               in lines if jid in x]) for jid in range(len(pts))]
            # Align the shared coordinate on the right
            anglesAll = []
            for v in lineCoordsByJunction:
                # Remove duplicates if they exist
                v = np.unique(v, axis=0)
                if len(v) < 2:
                    anglesAll.append([2 * np.pi, ])
                else:
                    sharedCoord = np.array(Counter([tuple(x) for x in np.vstack(
                        (v[:, 0:2], v[:, 2:4])).tolist()]).most_common(1)[0][0])
                    v = np.array([(x if np.all(x[0:2] != sharedCoord) else np.hstack((x[2:4], x[0:2]))) for x in v])
                    v = v[:, 0:2] - v[:, 2:4]
                    # Repeatedly find the closest neighbor (w.r.t. angle) until every line has 2 neighbors.
                    partners = np.full((v.shape[0], 2), -1, dtype=np.int64)
                    _c = 0
                    while np.sum(partners < 0) > 2:
                        _c += 1
                        if _c > 100:
                            _c = -1
                            break
                        mtxDist = np.full((v.shape[0], v.shape[0]), 1e6, dtype=np.float64)
                        a, aCounts = np.unique(partners.flatten(), return_counts=True)
                        candidates = [x for x in np.arange(v.shape[0]) if x not in a[aCounts >= 2]]
                        for i1 in candidates:
                            for i2 in candidates:
                                if i1 < i2 and partners[i1, 0] != i2 and partners[i1, 1] != i2 \
                                        and partners[i2, 0] != i1 and partners[i2, 1] != i1:
                                    mtxDist[i1, i2] = vecAngle(v[i1], v[i2])
                        i1, i2 = np.unravel_index(mtxDist.argmin(), mtxDist.shape)
                        partners[i1, 0 if partners[i1, 0] < 0 else 1] = i2
                        partners[i2, 0 if partners[i2, 0] < 0 else 1] = i1
                    # Compute angles between neighbors
                    if _c < 0:
                        anglesAll.append([])
                    else:
                        pairs = np.array([sorted([i, partners[i,0]]) for i in range(len(partners))] + \
                                         [sorted([i, partners[i,1]]) for i in range(len(partners))])
                        pairs = np.unique(pairs[np.all(pairs >= 0, axis=1)], axis=0)
                        angles = [vecAngle(v[p[0], :], v[p[1], :]) for p in pairs if p[0] >= 0 and p[1] >= 0]
                        angles.append( 2 * np.pi - np.sum(angles) )
                        if angles[-1] < 0:
                            raise Exception('Invalid angle encountered. This indicates a bug in the code.')
                        anglesAll.append(angles)
            # Store
            annotations.append(FrameAnnotations(fid, pts, lines, lineCoords,
                lineCoordsByJunction, anglesAll, lineLengths))
        return annotations
# Test
if __name__ == "__main__":
    import numpy as np
    filename = 'web_200hz-009.xyt.npy.txt'
    annotations = loadAnnotations(filename)

