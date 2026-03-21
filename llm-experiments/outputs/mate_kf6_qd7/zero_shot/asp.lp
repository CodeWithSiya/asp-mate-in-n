% Domain
file(1..8).
rank(1..8).

% Directions for pieces
rook_dir(1,0). rook_dir(-1,0). rook_dir(0,1). rook_dir(0,-1).
bishop_dir(1,1). bishop_dir(1,-1). bishop_dir(-1,1). bishop_dir(-1,-1).
queen_dir(1,0). queen_dir(-1,0). queen_dir(0,1). queen_dir(0,-1).
queen_dir(1,1). queen_dir(1,-1). queen_dir(-1,1). queen_dir(-1,-1).
king_dir(1,0). king_dir(-1,0). king_dir(0,1). king_dir(0,-1).
king_dir(1,1). king_dir(1,-1). king_dir(-1,1). king_dir(-1,-1).

% Opponent color
opponent(white, black).
opponent(black, white).

% Generate candidate moves for the side to move
% Queen moves (sliding)
candidate_move(C, queen, FX, FY, TX, TY) :-
    to_move(C),
    piece(C, queen, FX, FY),
    queen_dir(DX, DY),
    TX = FX + DX * S,
    TY = FY + DY * S,
    S = 1..8,
    file(TX), rank(TY),
    % All squares in between must be empty
    not blocked_queen(FX, FY, DX, DY, S).

% A square is blocked if any piece is on an intermediate square
blocked_queen(FX, FY, DX, DY, S) :-
    queen_dir(DX, DY),
    file(FX), rank(FY),
    S = 2..8,
    IX = FX + DX * (S-1),
    IY = FY + DY * (S-1),
    file(IX), rank(IY),
    piece(_, _, IX, IY).

blocked_queen(FX, FY, DX, DY, S) :-
    queen_dir(DX, DY),
    file(FX), rank(FY),
    S = 2..8,
    blocked_queen(FX, FY, DX, DY, S-1).

% Cannot capture own piece
:- candidate_move(C, queen, FX, FY, TX, TY), piece(C, _, TX, TY).

% King moves
candidate_move(C, king, FX, FY, TX, TY) :-
    to_move(C),
    king(C, FX, FY),
    king_dir(DX, DY),
    TX = FX + DX,
    TY = FY + DY,
    file(TX), rank(TY),
    not piece(C, _, TX, TY).

% After a candidate move, compute the resulting board state
% Piece positions after move (C, Type, FX, FY, TX, TY)
% The moved piece is at (TX, TY), captured piece (if any) is removed

% Check if opponent king is in check after move
% Square attacked by queen after move
queen_attacks_after(C, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    queen_dir(DX, DY),
    SX = MTX + DX * S,
    SY = MTY + DY * S,
    S = 1..8,
    file(SX), rank(SY),
    not blocked_after_queen(C, MType, MFX, MFY, MTX, MTY, DX, DY, S).

blocked_after_queen(C, MType, MFX, MFY, MTX, MTY, DX, DY, S) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    queen_dir(DX, DY),
    S = 2..8,
    IX = MTX + DX * (S-1),
    IY = MTY + DY * (S-1),
    file(IX), rank(IY),
    piece_after(C, MType, MFX, MFY, MTX, MTY, _, IX, IY).

blocked_after_queen(C, MType, MFX, MFY, MTX, MTY, DX, DY, S) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    queen_dir(DX, DY),
    S = 2..8,
    blocked_after_queen(C, MType, MFX, MFY, MTX, MTY, DX, DY, S-1).

% piece_after: pieces on board after move
piece_after(C, MType, MFX, MFY, MTX, MTY, PC, PX, PY) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    piece(PC, _, PX, PY),
    not (PX = MFX, PY = MFY),
    not (PX = MTX, PY = MTY).

piece_after(C, MType, MFX, MFY, MTX, MTY, C, MTX, MTY) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY).

% King attacks after move
king_attacks_after(C, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    king(C, KX, KY),
    king_dir(DX, DY),
    SX = KX + DX,
    SY = KY + DY,
    file(SX), rank(SY),
    not (KX = MFX, KY = MFY).

% Opponent king position after move
opp_king_after(C, MType, MFX, MFY, MTX, MTY, KX, KY) :-
    candidate_move(C, MType, MFX, MFY, MTX, MTY),
    opponent(C, OC),
    king(OC, KX, KY),
    not (KX = MTX, KY = MTY).

% Check if opponent king is in check after move
in_check_after(C, MType, MFX, MFY, MTX, MTY) :-
    opp_king_after(C, MType, MFX, MFY, MTX, MTY, KX, KY),
    MType = queen,
    queen_attacks_after(C, MType, MFX, MFY, MTX, MTY, KX, KY).

in_check_after(C, MType, MFX, MFY, MTX, MTY) :-
    opp_king_after(C, MType, MFX, MFY, MTX, MTY, KX, KY),
    MType = king,
    king_attacks_after(C, MType, MFX, MFY, MTX, MTY, KX, KY).

% Check if opponent king can escape: for each possible king move
% Opponent king's escape squares
opp_king_escape(C, MType, MFX, MFY, MTX, MTY, EX, EY) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    opponent(C, OC),
    opp_king_after(C, MType, MFX, MFY, MTX, MTY, KX, KY),
    king_dir(DX, DY),
    EX = KX + DX,
    EY = KY + DY,
    file(EX), rank(EY),
    not piece_after(C, MType, MFX, MFY, MTX, MTY, OC, EX, EY),
    not still_attacked(C, MType, MFX, MFY, MTX, MTY, EX, EY).

% A square is still attacked by moving side after the move
still_attacked(C, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    MType = queen,
    queen_attacks_after(C, MType, MFX, MFY, MTX, MTY, SX, SY).

still_attacked(C, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    king_attacks_after(C, MType, MFX, MFY, MTX, MTY, SX, SY).

% Also check if escape square is occupied by own piece (already handled above)
% Check if any piece of moving side (after move) attacks escape square
% Queen after move attacks escape
still_attacked(C, MType, MFX, MFY, MTX, MTY, SX, SY) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    piece_after(C, MType, MFX, MFY, MTX, MTY, C, QX, QY),
    piece(C, queen, QX, QY),
    queen_dir(DX, DY),
    SX = QX + DX * S,
    SY = QY + DY * S,
    S = 1..8,
    file(SX), rank(SY),
    not blocked_after_queen2(C, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, S).

blocked_after_queen2(C, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, S) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    queen_dir(DX, DY),
    S = 2..8,
    IX = QX + DX * (S-1),
    IY = QY + DY * (S-1),
    file(IX), rank(IY),
    piece_after(C, MType, MFX, MFY, MTX, MTY, _, IX, IY).

blocked_after_queen2(C, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, S) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    queen_dir(DX, DY),
    S = 2..8,
    blocked_after_queen2(C, MType, MFX, MFY, MTX, MTY, QX, QY, DX, DY, S-1).

% Checkmate: in check and no escape
checkmate_after(C, MType, MFX, MFY, MTX, MTY) :-
    in_check_after(C, MType, MFX, MFY, MTX, MTY),
    not opp_king_escape(C, MType, MFX, MFY, MTX, MTY, _, _).

% The mating move
mate_move(C, MType, MFX, MFY, MTX, MTY) :-
    checkmate_after(C, MType, MFX, MFY, MTX, MTY).

#show mate_move/6.