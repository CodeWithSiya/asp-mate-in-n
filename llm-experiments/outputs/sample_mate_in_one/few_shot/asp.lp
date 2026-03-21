I'll analyze the position and write a specialized ASP program to find the mate in one.

Position analysis:
- White: King g1(7,1), Queen h1(8,1), Pawns f2(6,2), g2(7,2), h2(8,2)
- Black: King g8(7,8), Pawns f7(6,7), g7(7,7), h7(8,7)
- White to move

The queen on h1 can potentially deliver mate. Let me think about queen moves and check/mate detection.

% Board coordinates
coord(1..8).
square(X,Y) :- coord(X), coord(Y).

% Input facts (already provided, repeated for completeness in reasoning)
% piece/king/to_move facts are given as input

% ============================================================
% Occupied squares
% ============================================================
occupied(X,Y,Color,Type) :- piece(Color,Type,X,Y).
occupied_by(X,Y,Color) :- piece(Color,_,X,Y).

% ============================================================
% Queen sliding move directions
% ============================================================
dir(1,0). dir(-1,0). dir(0,1). dir(0,-1).
dir(1,1). dir(1,-1). dir(-1,1). dir(-1,-1).

% Squares a queen on (FX,FY) can reach along direction (DX,DY) step by step
% We enumerate up to 7 steps
step(1). step(2). step(3). step(4). step(5). step(6). step(7).

% A queen slide is blocked if any piece is on an intermediate square
intermediate(FX,FY,DX,DY,S,TX,TY) :-
    dir(DX,DY),
    step(S), S > 1,
    TX = FX + DX*S,
    TY = FY + DY*S,
    square(TX,TY),
    piece(white,queen,FX,FY),
    IS = S - 1, IS >= 1,
    IX = FX + DX*IS,
    IY = FY + DY*IS,
    square(IX,IY),
    occupied_by(IX,IY,_).

% Queen can reach (TX,TY) from (FX,FY) in direction (DX,DY) at step S
% if no intermediate square is blocked
queen_reach(FX,FY,DX,DY,S,TX,TY) :-
    dir(DX,DY),
    step(S),
    TX = FX + DX*S,
    TY = FY + DY*S,
    square(TX,TY),
    piece(white,queen,FX,FY),
    not intermediate(FX,FY,DX,DY,S,TX,TY).

% But we also need to ensure all steps < S are clear (not blocked before reaching TX,TY)
% Actually let's redo: queen can reach step S only if step S-1 is not occupied (for S>1)

% Clear path: no piece on squares at steps 1..S-1
path_clear(FX,FY,DX,DY,1) :-
    dir(DX,DY),
    piece(white,queen,FX,FY),
    X1 = FX+DX, Y1 = FY+DY,
    square(X1,Y1).

path_clear(FX,FY,DX,DY,S) :-
    dir(DX,DY),
    piece(white,queen,FX,FY),
    step(S), S > 1,
    path_clear(FX,FY,DX,DY,S-1),
    PS = S-1,
    IX = FX + DX*PS,
    IY = FY + DY*PS,
    not occupied_by(IX,IY,_).

% Queen move to empty square
queen_move(FX,FY,TX,TY) :-
    dir(DX,DY),
    step(S),
    piece(white,queen,FX,FY),
    TX = FX + DX*S,
    TY = FY + DY*S,
    square(TX,TY),
    path_clear(FX,FY,DX,DY,S),
    not occupied_by(TX,TY,_).

% Queen capture (can capture black piece)
queen_capture(FX,FY,TX,TY) :-
    dir(DX,DY),
    step(S),
    piece(white,queen,FX,FY),
    TX = FX + DX*S,
    TY = FY + DY*S,
    square(TX,TY),
    path_clear(FX,FY,DX,DY,S),
    occupied_by(TX,TY,black).

% ============================================================
% White king moves
% ============================================================
king_dir(DX,DY) :- DX = -1, DY = -1.
king_dir(DX,DY) :- DX = -1, DY = 0.
king_dir(DX,DY) :- DX = -1, DY = 1.
king_dir(DX,DY) :- DX = 0,  DY = -1.
king_dir(DX,DY) :- DX = 0,  DY = 1.
king_dir(DX,DY) :- DX = 1,  DY = -1.
king_dir(DX,DY) :- DX = 1,  DY = 0.
king_dir(DX,DY) :- DX = 1,  DY = 1.

white_king_move(FX,FY,TX,TY) :-
    king(white,FX,FY),
    king_dir(DX,DY),
    TX = FX+DX, TY = FY+DY,
    square(TX,TY),
    not occupied_by(TX,TY,white).

% ============================================================
% White pawn moves (white pawns move +Y direction)
% ============================================================
white_pawn_move(FX,FY,FX,FY+1) :-
    piece(white,pawn,FX,FY),
    FY+1 <= 8,
    not occupied_by(FX,FY+1,_).

white_pawn_capture(FX,FY,FX+1,FY+1) :-
    piece(white,pawn,FX,FY),
    FY+1 <= 8, FX+1 <= 8,
    occupied_by(FX+1,FY+1,black).

white_pawn_capture(FX,FY,FX-1,FY+1) :-
    piece(white,pawn,FX,FY),
    FY+1 <= 8, FX-1 >= 1,
    occupied_by(FX-1,FY+1,black).

% ============================================================
% All white candidate moves (from, to, piece type, capture flag)
% ============================================================
white_move(queen,FX,FY,TX,TY,no_capture) :- queen_move(FX,FY,TX,TY).
white_move(queen,FX,FY,TX,TY,capture)    :- queen_capture(FX,FY,TX,TY).
white_move(king,FX,FY,TX,TY,no_capture)  :- white_king_move(FX,FY,TX,TY), not occupied_by(TX,TY,black).
white_move(king,FX,FY,TX,TY,capture)     :- white_king_move(FX,FY,TX,TY), occupied_by(TX,TY,black).
white_move(pawn,FX,FY,TX,TY,no_capture)  :- white_pawn_move(FX,FY,TX,TY).
white_move(pawn,FX,FY,TX,TY,capture)     :- white_pawn_capture(FX,FY,TX,TY).

% ============================================================
% After white move: compute resulting board state
% White piece moves from (FX,FY) to (TX,TY), possibly capturing black piece at (TX,TY)
% ============================================================

% Pieces remaining after white move (PType,FX,FY) -> (TX,TY)
% White pieces after move
white_piece_after(PT2,X2,Y2, PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    piece(white,PT2,X2,Y2),
    % The moved piece is now at TX,TY
    % Other white pieces stay
    not (PT2 = PT, X2 = FX, Y2 = FY).

% Actually let's track positions differently
% After move (PT,FX,FY,TX,TY):
% White king position
wking_after(TX,TY,PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    PT = king.

wking_after(KX,KY,PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    PT != king,
    king(white,KX,KY).

% ============================================================
% After white move, does white king end up in check?
% We need to check if black pieces attack the white king square.
% Black has only a king and pawns.
% ============================================================

% Black king attacks
bking_attacks(X,Y) :-
    king(black,KX,KY),
    king_dir(DX,DY),
    X = KX+DX, Y = KY+DY,
    square(X,Y).

% Black pawn attacks (black pawns move -Y, attack diagonally -Y)
bpawn_attacks(X,Y) :-
    piece(black,pawn,PX,PY),
    X = PX+1, Y = PY-1,
    square(X,Y).

bpawn_attacks(X,Y) :-
    piece(black,pawn,PX,PY),
    X = PX-1, Y = PY-1,
    square(X,Y).

% White king in check after move (PT,FX,FY,TX,TY)?
% (We ignore that black pawn might be captured by white move - simplified for this position)
wking_in_check_after(PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    wking_after(WKX,WKY,PT,FX,FY,TX,TY),
    bking_attacks(WKX,WKY).

wking_in_check_after(PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    wking_after(WKX,WKY,PT,FX,FY,TX,TY),
    bpawn_attacks(WKX,WKY).

% Legal white move: doesn't leave white king in check
legal_white_move(PT,FX,FY,TX,TY) :-
    white_move(PT,FX,FY,TX,TY,_),
    not wking_in_check_after(PT,FX,FY,TX,TY).

% ============================================================
% After a legal white move, compute the resulting position
% and check if black king is in check
% ============================================================

% Position of white queen after move
wqueen_pos_after(TX,TY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    PT = queen.

wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    PT != queen,
    piece(white,queen,QX,QY).

% White king position after legal move
wking_pos_after(TX,TY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    PT = king.

wking_pos_after(KX,KY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    PT != king,
    king(white,KX,KY).

% White pawn positions after move
wpawn_pos_after(TX,TY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    PT = pawn.

wpawn_pos_after(PX,PY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    piece(white,pawn,PX,PY),
    not (PT = pawn, PX = FX, PY = FY).

% ============================================================
% Black pieces after white move
% Black pieces: king at (7,8), pawns at (6,7),(7,7),(8,7)
% A black piece is removed if white captured it (TX,TY = its position)
% ============================================================

% Black king still present after white move?
bking_present_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    not (TX = BKX, TY = BKY).

% Black pawn present after white move
bpawn_present_after(PX,PY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    piece(black,pawn,PX,PY),
    not (TX = PX, TY = PY).

% ============================================================
% Does white queen (at QX,QY) attack black king (at BKX,BKY)
% given the board after white's move?
% Need to check sliding path is clear.
% ============================================================

% Helper: sign of difference
sign_val(D,1)  :- D > 0.
sign_val(D,-1) :- D < 0.
sign_val(0,0).

% Queen attacks king if they share rank/file/diagonal and path is clear
% After move (PT,FX,FY,TX,TY):

% Check if queen at (QX,QY) attacks (BKX,BKY) after white move
% Same rank
queen_attacks_king_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    bking_present_after(PT,FX,FY,TX,TY),
    QY = BKY,
    QX != BKX,
    queen_rank_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

queen_attacks_king_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    bking_present_after(PT,FX,FY,TX,TY),
    QX = BKX,
    QY != BKY,
    queen_file_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

queen_attacks_king_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    bking_present_after(PT,FX,FY,TX,TY),
    DX = BKX - QX, DY = BKY - QY,
    DX != 0, DY != 0,
    DX = DY,
    queen_diag1_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

queen_attacks_king_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    bking_present_after(PT,FX,FY,TX,TY),
    DX = BKX - QX, DY = BKY - QY,
    DX != 0, DY != 0,
    DX = -DY,
    queen_diag2_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

% ============================================================
% Path clear checks for queen attacks after white move
% We need to check no piece is between queen and black king
% ============================================================

% For this specific position, let's enumerate squares between queen and king
% and check they are not occupied after the move.

% A square (IX,IY) is between (QX,QY) and (BKX,BKY) on same rank if QY=BKY
% and IX is strictly between QX and BKX, IY=QY

% Occupied after white move (by any piece)
occupied_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wpawn_pos_after(X,Y,PT,FX,FY,TX,TY).

occupied_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wking_pos_after(X,Y,PT,FX,FY,TX,TY).

occupied_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(X,Y,PT,FX,FY,TX,TY).

occupied_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(X,Y,PT,FX,FY,TX,TY).

% Black king square is occupied by black king (if present)
occupied_after(BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    bking_present_after(PT,FX,FY,TX,TY).

% Between squares on same rank (QY=BKY, QX < BKX)
between_rank(QX,QY,BKX,IX) :-
    coord(QX), coord(BKX), coord(IX),
    QX < BKX,
    IX > QX, IX < BKX,
    QY = QY. % trivially true, just for structure

% Let's use a simpler approach: check if any piece is between queen and king

% Same rank path clear: no piece between QX and BKX on row QY
queen_rank_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    QY = BKY,
    QX < BKX,
    not queen_rank_blocked_after(QX,QY,BKX,PT,FX,FY,TX,TY).

queen_rank_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    QY = BKY,
    QX > BKX,
    not queen_rank_blocked_after2(QX,QY,BKX,PT,FX,FY,TX,TY).

queen_rank_blocked_after(QX,QY,BKX,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX),
    IX > QX, IX < BKX,
    occupied_after(IX,QY,PT,FX,FY,TX,TY).

queen_rank_blocked_after2(QX,QY,BKX,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX),
    IX < QX, IX > BKX,
    occupied_after(IX,QY,PT,FX,FY,TX,TY).

% Same file path clear
queen_file_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    QX = BKX,
    QY < BKY,
    not queen_file_blocked_after(QX,QY,BKY,PT,FX,FY,TX,TY).

queen_file_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    QX = BKX,
    QY > BKY,
    not queen_file_blocked_after2(QX,QY,BKY,PT,FX,FY,TX,TY).

queen_file_blocked_after(QX,QY,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IY),
    IY > QY, IY < BKY,
    occupied_after(QX,IY,PT,FX,FY,TX,TY).

queen_file_blocked_after2(QX,QY,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IY),
    IY < QY, IY > BKY,
    occupied_after(QX,IY,PT,FX,FY,TX,TY).

% Diagonal path clear (DX=DY, i.e., same diagonal direction)
queen_diag1_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    DX = BKX - QX, DY = BKY - QY,
    DX = DY, DX != 0,
    not queen_diag1_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

queen_diag1_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    IX = IY - QY + QX,  % on same diagonal
    DX = BKX - QX,
    DX > 0,
    IX > QX, IX < BKX,
    IY = IX - QX + QY,
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

queen_diag1_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX - QX,
    DX < 0,
    IX < QX, IX > BKX,
    IY = IX - QX + QY,
    square(IX,IY),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

queen_diag2_path_clear_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    DX = BKX - QX, DY = BKY - QY,
    DX = -DY, DX != 0,
    not queen_diag2_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY).

queen_diag2_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX - QX,
    DX > 0,
    IX > QX, IX < BKX,
    IY = QY - (IX - QX),
    square(IX,IY),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

queen_diag2_blocked_after(QX,QY,BKX,BKY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX - QX,
    DX < 0,
    IX < QX, IX > BKX,
    IY = QY - (IX - QX),
    square(IX,IY),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

% ============================================================
% White king attacks (for checking if white king gives check - unlikely but complete)
% ============================================================
wking_attacks_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wking_pos_after(KX,KY,PT,FX,FY,TX,TY),
    king_dir(DX,DY),
    X = KX+DX, Y = KY+DY,
    square(X,Y).

% White pawn attacks after move
wpawn_attacks_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wpawn_pos_after(PX,PY,PT,FX,FY,TX,TY),
    X = PX+1, Y = PY+1,
    square(X,Y).

wpawn_attacks_after(X,Y,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wpawn_pos_after(PX,PY,PT,FX,FY,TX,TY),
    X = PX-1, Y = PY+1,
    square(X,Y).

% ============================================================
% Black king in check after white move
% ============================================================
bking_in_check_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    queen_attacks_king_after(PT,FX,FY,TX,TY).

bking_in_check_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    wking_attacks_after(BKX,BKY,PT,FX,FY,TX,TY).

bking_in_check_after(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    wpawn_attacks_after(BKX,BKY,PT,FX,FY,TX,TY).

% ============================================================
% Black king escape moves after white move (PT,FX,FY,TX,TY)
% Black king can move to adjacent squares not occupied by black pieces
% and not attacked by white after the move
% ============================================================

% Black king position (it's at 7,8 and not captured)
bking_move_after(BKX,BKY,NX,NY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bking_present_after(PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    king_dir(DX,DY),
    NX = BKX+DX, NY = BKY+DY,
    square(NX,NY),
    % Not occupied by black piece
    not bpawn_present_after(NX,NY,PT,FX,FY,TX,TY),
    % Not the black king's own square (already handled by king_dir != 0,0)
    % Not occupied by white piece after move (except we can capture white pieces)
    % Actually black king can capture white pieces
    % But can't move to square attacked by white
    not white_attacks_square_after(NX,NY,PT,FX,FY,TX,TY).

% White attacks a square after move (PT,FX,FY,TX,TY)
% Queen attacks
white_attacks_square_after(SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    queen_attacks_square_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

% King attacks
white_attacks_square_after(SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wking_attacks_after(SX,SY,PT,FX,FY,TX,TY).

% Pawn attacks
white_attacks_square_after(SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    wpawn_attacks_after(SX,SY,PT,FX,FY,TX,TY).

% Queen attacks a specific square (SX,SY) from (QX,QY)
% Same rank
queen_attacks_square_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QY = SY, QX != SX,
    queen_rank_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

% Same file
queen_attacks_square_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QX = SX, QY != SY,
    queen_file_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

% Diagonal DX=DY
queen_attacks_square_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX, DY = SY - QY,
    DX != 0, DY != 0, DX = DY,
    queen_diag1_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

% Anti-diagonal DX=-DY
queen_attacks_square_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX, DY = SY - QY,
    DX != 0, DY != 0, DX = -DY,
    queen_diag2_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

% Path clear helpers for queen to square (SX,SY)
% These are similar to the king attack path clears but for arbitrary target squares

% Rank path clear (QY=SY)
queen_rank_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QY = SY, QX < SX,
    not queen_rank_blocked_to_after(QX,QY,SX,PT,FX,FY,TX,TY).

queen_rank_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QY = SY, QX > SX,
    not queen_rank_blocked_to_after2(QX,QY,SX,PT,FX,FY,TX,TY).

queen_rank_blocked_to_after(QX,QY,SX,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX),
    IX > QX, IX < SX,
    % Exclude the black king square from blocking (we're checking attack, king can be captured)
    not (king(black,IX,QY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,QY,PT,FX,FY,TX,TY).

queen_rank_blocked_to_after2(QX,QY,SX,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX),
    IX < QX, IX > SX,
    not (king(black,IX,QY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,QY,PT,FX,FY,TX,TY).

% File path clear (QX=SX)
queen_file_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QX = SX, QY < SY,
    not queen_file_blocked_to_after(QX,QY,SY,PT,FX,FY,TX,TY).

queen_file_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    QX = SX, QY > SY,
    not queen_file_blocked_to_after2(QX,QY,SY,PT,FX,FY,TX,TY).

queen_file_blocked_to_after(QX,QY,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IY),
    IY > QY, IY < SY,
    not (king(black,QX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(QX,IY,PT,FX,FY,TX,TY).

queen_file_blocked_to_after2(QX,QY,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IY),
    IY < QY, IY > SY,
    not (king(black,QX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(QX,IY,PT,FX,FY,TX,TY).

% Diagonal path clear (DX=DY)
queen_diag1_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX,
    DX > 0,
    not queen_diag1_blocked_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

queen_diag1_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX,
    DX < 0,
    not queen_diag1_blocked_to_after2(QX,QY,SX,SY,PT,FX,FY,TX,TY).

queen_diag1_blocked_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = SX - QX, DX > 0,
    IX > QX, IX < SX,
    IY = QY + (IX - QX),
    square(IX,IY),
    not (king(black,IX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

queen_diag1_blocked_to_after2(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = SX - QX, DX < 0,
    IX < QX, IX > SX,
    IY = QY + (IX - QX),
    square(IX,IY),
    not (king(black,IX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

% Anti-diagonal path clear (DX=-DY)
queen_diag2_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX,
    DX > 0,
    not queen_diag2_blocked_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY).

queen_diag2_path_clear_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    DX = SX - QX,
    DX < 0,
    not queen_diag2_blocked_to_after2(QX,QY,SX,SY,PT,FX,FY,TX,TY).

queen_diag2_blocked_to_after(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = SX - QX, DX > 0,
    IX > QX, IX < SX,
    IY = QY - (IX - QX),
    square(IX,IY),
    not (king(black,IX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

queen_diag2_blocked_to_after2(QX,QY,SX,SY,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = SX - QX, DX < 0,
    IX < QX, IX > SX,
    IY = QY - (IX - QX),
    square(IX,IY),
    not (king(black,IX,IY), bking_present_after(PT,FX,FY,TX,TY)),
    occupied_after(IX,IY,PT,FX,FY,TX,TY).

% ============================================================
% Black has a legal reply if black king can move somewhere safe
% or if black can block/capture (but black only has pawns and king here)
% Black pawn moves (black pawns move -Y direction)
% ============================================================

% Black pawn move after white move
bpawn_move_after(PX,PY,PX,PY-1,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(PX,PY,PT,FX,FY,TX,TY),
    PY-1 >= 1,
    not occupied_after(PX,PY-1,PT,FX,FY,TX,TY).

% Black pawn capture
bpawn_capture_after(PX,PY,PX+1,PY-1,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(PX,PY,PT,FX,FY,TX,TY),
    PY-1 >= 1, PX+1 <= 8,
    occupied_after(PX+1,PY-1,PT,FX,FY,TX,TY),
    % Must be white piece
    wpawn_pos_after(PX+1,PY-1,PT,FX,FY,TX,TY).

bpawn_capture_after(PX,PY,PX-1,PY-1,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(PX,PY,PT,FX,FY,TX,TY),
    PY-1 >= 1, PX-1 >= 1,
    occupied_after(PX-1,PY-1,PT,FX,FY,TX,TY),
    wpawn_pos_after(PX-1,PY-1,PT,FX,FY,TX,TY).

% Also black pawn can capture white queen
bpawn_capture_after(PX,PY,PX+1,PY-1,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(PX,PY,PT,FX,FY,TX,TY),
    PY-1 >= 1, PX+1 <= 8,
    wqueen_pos_after(PX+1,PY-1,PT,FX,FY,TX,TY).

bpawn_capture_after(PX,PY,PX-1,PY-1,PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bpawn_present_after(PX,PY,PT,FX,FY,TX,TY),
    PY-1 >= 1, PX-1 >= 1,
    wqueen_pos_after(PX-1,PY-1,PT,FX,FY,TX,TY).

% ============================================================
% After black's reply, is black king still in check?
% We need to verify that black's reply doesn't escape check.
% For mate in one: after white's move, black has NO legal reply.
% A legal black reply is one where black king is NOT in check after.
% ============================================================

% Black king escape: king moves to (NX,NY)
% This is a legal reply if the square is not attacked by white
% (already computed in bking_move_after)

% Black has a legal reply after white move (PT,FX,FY,TX,TY) if:
% 1. Black king can move to a safe square
black_has_reply(PT,FX,FY,TX,TY) :-
    bking_move_after(BKX,BKY,NX,NY,PT,FX,FY,TX,TY),
    % The destination is not occupied by a black piece (already checked)
    % and not attacked by white (already checked)
    % Also need: black king doesn't move into check from white pieces
    % (already handled by white_attacks_square_after)
    % But we also need to ensure the new square isn't attacked considering
    % the black king has moved (so the old king square is now empty)
    % This is complex - for simplicity in this specific position, let's trust the computation
    square(NX,NY).

% 2. Black pawn can block or capture to remove check
% After black pawn move, is black king still in check?
% This requires recomputing white attacks on black king after black's pawn move.
% For this specific position, let's check if a pawn move/capture removes check.

% After white move (PT,FX,FY,TX,TY) and black pawn moves from (PX,PY) to (NPX,NPY):
% Does this block the check?

% White queen attacks black king - can a pawn block?
% This is complex. Let's handle it: after black pawn move, recheck if white queen still attacks king.

% For the specific position, black pawns are at f7,g7,h7 and king at g8.
% Let's check if any black pawn move removes the check.

black_has_pawn_reply(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bking_in_check_after(PT,FX,FY,TX,TY),
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    % After this pawn move, is black king still in check?
    not bking_still_in_check_after_pawn_move(PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

black_has_pawn_capture_reply(PT,FX,FY,TX,TY) :-
    legal_white_move(PT,FX,FY,TX,TY),
    bking_in_check_after(PT,FX,FY,TX,TY),
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    not bking_still_in_check_after_pawn_capture(PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

% After black pawn moves from (PX,PY) to (NPX,NPY), is white queen still attacking black king?
% New occupied squares: remove (PX,PY), add (NPX,NPY) for black pawn

occupied_after_pawn_move(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    occupied_after(IX,IY,PT,FX,FY,TX,TY),
    not (IX = PX, IY = PY).  % pawn left its square

occupied_after_pawn_move(NPX,NPY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY).

% After pawn capture, the captured white piece is removed
occupied_after_pawn_capture(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    occupied_after(IX,IY,PT,FX,FY,TX,TY),
    not (IX = PX, IY = PY),   % pawn left
    not (IX = NPX, IY = NPY). % captured piece removed

occupied_after_pawn_capture(NPX,NPY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY).

% Is black king still in check after pawn move?
bking_still_in_check_after_pawn_move(PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    queen_attacks_king_after_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

bking_still_in_check_after_pawn_capture(PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    king(black,BKX,BKY),
    wqueen_pos_after(QX,QY,PT,FX,FY,TX,TY),
    % If pawn captured the queen, queen no longer attacks
    not (NPX = QX, NPY = QY),
    queen_attacks_king_after_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

% Queen attacks king after pawn move (same rank/file/diagonal checks with new occupancy)
% Same file (most likely for this position: queen on h-file attacking king on g8 via file?)
% Let's do rank and file and diagonal

queen_attacks_king_after_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY = BKY, QX != BKX,
    queen_rank_clear_pawn_move(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX = BKX, QY != BKY,
    queen_file_clear_pawn_move(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DY = BKY-QY,
    DX != 0, DY != 0, DX = DY,
    queen_diag1_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DY = BKY-QY,
    DX != 0, DY != 0, DX = -DY,
    queen_diag2_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_rank_clear_pawn_move(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX < BKX,
    not queen_rank_blocked_pm(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,left).

queen_rank_clear_pawn_move(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX > BKX,
    not queen_rank_blocked_pm(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,right).

queen_rank_blocked_pm(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,left) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), QX < BKX,
    IX > QX, IX < BKX,
    occupied_after_pawn_move(IX,QY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_rank_blocked_pm(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,right) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), QX > BKX,
    IX < QX, IX > BKX,
    occupied_after_pawn_move(IX,QY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_file_clear_pawn_move(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY < BKY,
    not queen_file_blocked_pm(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,up).

queen_file_clear_pawn_move(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY > BKY,
    not queen_file_blocked_pm(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,down).

queen_file_blocked_pm(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,up) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IY), QY < BKY,
    IY > QY, IY < BKY,
    occupied_after_pawn_move(QX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_file_blocked_pm(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,down) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IY), QY > BKY,
    IY < QY, IY > BKY,
    occupied_after_pawn_move(QX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag1_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX > 0,
    not queen_diag1_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos).

queen_diag1_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX < 0,
    not queen_diag1_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg).

queen_diag1_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX > 0,
    IX > QX, IX < BKX,
    IY = QY + (IX-QX),
    square(IX,IY),
    occupied_after_pawn_move(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag1_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX < 0,
    IX < QX, IX > BKX,
    IY = QY + (IX-QX),
    square(IX,IY),
    occupied_after_pawn_move(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag2_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX > 0,
    not queen_diag2_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos).

queen_diag2_clear_pawn_move(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX < 0,
    not queen_diag2_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg).

queen_diag2_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX > 0,
    IX > QX, IX < BKX,
    IY = QY - (IX-QX),
    square(IX,IY),
    occupied_after_pawn_move(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag2_blocked_pm(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg) :-
    bpawn_move_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX < 0,
    IX < QX, IX > BKX,
    IY = QY - (IX-QX),
    square(IX,IY),
    occupied_after_pawn_move(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

% Same for pawn capture
queen_attacks_king_after_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY = BKY, QX != BKX,
    queen_rank_clear_pawn_capture(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX = BKX, QY != BKY,
    queen_file_clear_pawn_capture(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DY = BKY-QY,
    DX != 0, DY != 0, DX = DY,
    queen_diag1_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_attacks_king_after_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DY = BKY-QY,
    DX != 0, DY != 0, DX = -DY,
    queen_diag2_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_rank_clear_pawn_capture(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX < BKX,
    not queen_rank_blocked_pc(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,left).

queen_rank_clear_pawn_capture(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QX > BKX,
    not queen_rank_blocked_pc(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,right).

queen_rank_blocked_pc(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,left) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), QX < BKX,
    IX > QX, IX < BKX,
    occupied_after_pawn_capture(IX,QY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_rank_blocked_pc(QX,QY,BKX,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,right) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), QX > BKX,
    IX < QX, IX > BKX,
    occupied_after_pawn_capture(IX,QY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_file_clear_pawn_capture(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY < BKY,
    not queen_file_blocked_pc(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,up).

queen_file_clear_pawn_capture(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    QY > BKY,
    not queen_file_blocked_pc(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,down).

queen_file_blocked_pc(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,up) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IY), QY < BKY,
    IY > QY, IY < BKY,
    occupied_after_pawn_capture(QX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_file_blocked_pc(QX,QY,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,down) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IY), QY > BKY,
    IY < QY, IY > BKY,
    occupied_after_pawn_capture(QX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag1_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX > 0,
    not queen_diag1_blocked_pc(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos).

queen_diag1_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX < 0,
    not queen_diag1_blocked_pc(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg).

queen_diag1_blocked_pc(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX > 0,
    IX > QX, IX < BKX,
    IY = QY + (IX-QX),
    square(IX,IY),
    occupied_after_pawn_capture(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag1_blocked_pc(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,neg) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    coord(IX), coord(IY),
    DX = BKX-QX, DX < 0,
    IX < QX, IX > BKX,
    IY = QY + (IX-QX),
    square(IX,IY),
    occupied_after_pawn_capture(IX,IY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY).

queen_diag2_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY) :-
    bpawn_capture_after(PX,PY,NPX,NPY,PT,FX,FY,TX,TY),
    DX = BKX-QX, DX > 0,
    not queen_diag2_blocked_pc(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY,pos).

queen_diag2_clear_pawn_capture(QX,QY,BKX,BKY,PT,FX,FY,TX,TY,PX,PY,NPX,NPY