% Coordinate domain
file(1..8).
rank(1..8).

% Occupied squares (any piece or king)
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

occupied_by_own(Color, X, Y) :- piece(Color, _, X, Y).
occupied_by_own(Color, X, Y) :- king(Color, X, Y).

% -------------------------------------------------------
% Queen move generation
% -------------------------------------------------------

% Horizontal blocking
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX > TX, BX < FX, BX > TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX > TX, BX < FX, BX > TX.

% Vertical blocking
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY > TY, BY < FY, BY > TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY > TY, BY < FY, BY > TY.

% Diagonal blocking (both diagonals)
% For diagonal moves: |TX-FX| = |TY-FY|, same sign pattern
% We need to check if any piece lies strictly between (FX,FY) and (TX,TY) on the diagonal

% Helper: sign of difference
pos_dir(1) :- file(1).
pos_dir(-1) :- file(1).

% Diagonal blocking: piece at (BX,BY) blocks diagonal from (FX,FY) to (TX,TY)
% BX is strictly between FX and TX (same direction), BY strictly between FY and TY
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX < TX, FY < TY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX < TX, FY < TY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.

blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY > TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY > TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX < TX, FY > TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX < TX, FY > TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.

blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY < TY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY < TY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.

% Queen can move to (TX,TY) from (FX,FY) if:
% same rank, not blocked horizontally, not own piece at target
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FY = TY,
    FX != TX,
    not blocked_h(FX, FY, TX),
    not occupied_by_own(Color, TX, TY).

% same file, not blocked vertically
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FX = TX,
    FY != TY,
    not blocked_v(FX, FY, TY),
    not occupied_by_own(Color, TX, TY).

% diagonal (up-right)
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FX < TX, FY < TY,
    TX - FX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

% diagonal (down-left)
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FX > TX, FY > TY,
    FX - TX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

% diagonal (up-left)
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FX > TX, FY < TY,
    FX - TX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

% diagonal (down-right)
queen_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    file(TX), rank(TY),
    FX < TX, FY > TY,
    TX - FX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% Rook move generation
% -------------------------------------------------------
rook_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    file(TX), rank(TY),
    FY = TY, FX != TX,
    not blocked_h(FX, FY, TX),
    not occupied_by_own(Color, TX, TY).

rook_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    file(TX), rank(TY),
    FX = TX, FY != TY,
    not blocked_v(FX, FY, TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% Bishop move generation
% -------------------------------------------------------
bishop_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    file(TX), rank(TY),
    FX < TX, FY < TY, TX - FX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

bishop_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    file(TX), rank(TY),
    FX > TX, FY > TY, FX - TX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

bishop_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    file(TX), rank(TY),
    FX > TX, FY < TY, FX - TX = TY - FY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

bishop_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    file(TX), rank(TY),
    FX < TX, FY > TY, TX - FX = FY - TY,
    not blocked_d(FX, FY, TX, TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% Knight move generation
% -------------------------------------------------------
knight_delta(1,2). knight_delta(2,1). knight_delta(2,-1). knight_delta(1,-2).
knight_delta(-1,-2). knight_delta(-2,-1). knight_delta(-2,1). knight_delta(-1,2).

knight_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    knight_delta(DX, DY),
    TX = FX + DX, TY = FY + DY,
    file(TX), rank(TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% Pawn move generation
% -------------------------------------------------------
pawn_dir(white, 1).
pawn_dir(black, -1).

% Single push
pawn_can_move(Color, FX, FY, FX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_dir(Color, D),
    TY = FY + D,
    rank(TY),
    not occupied(FX, TY).

% Double push from starting rank
pawn_start_rank(white, 2).
pawn_start_rank(black, 7).

pawn_can_move(Color, FX, FY, FX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_dir(Color, D),
    pawn_start_rank(Color, FY),
    TY = FY + 2*D,
    rank(TY),
    MidY = FY + D,
    not occupied(FX, MidY),
    not occupied(FX, TY).

% Pawn capture
pawn_can_move(Color, FX, FY, TX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_dir(Color, D),
    TY = FY + D,
    rank(TY),
    (TX = FX + 1 ; TX = FX - 1),
    file(TX),
    occupied(TX, TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% King move generation
% -------------------------------------------------------
king_delta(-1,-1). king_delta(-1,0). king_delta(-1,1).
king_delta(0,-1).                    king_delta(0,1).
king_delta(1,-1).  king_delta(1,0).  king_delta(1,1).

king_can_move(Color, FX, FY, TX, TY) :-
    king(Color, FX, FY),
    king_delta(DX, DY),
    TX = FX + DX, TY = FY + DY,
    file(TX), rank(TY),
    not occupied_by_own(Color, TX, TY).

% -------------------------------------------------------
% After a move, compute the new board state
% We need to check if the opponent king is in check
% and if the opponent has no legal moves
% -------------------------------------------------------

% Opponent color
opponent(white, black).
opponent(black, white).

% -------------------------------------------------------
% Check detection: is square (KX,KY) attacked by Color
% after a specific move has been made?
% We parameterize by the move: (MFX,MFY)->(MTX,MTY), piece moved is (MColor,MType)
% -------------------------------------------------------

% After move (MFX,MFY)->(MTX,MTY) by MColor:
% A piece of MColor at (PX,PY) exists if:
%   original piece at (PX,PY) AND NOT (PX=MFX,PY=MFY) AND NOT captured at (MTX,MTY) for opponent
%   OR (PX=MTX,PY=MTY) for the moved piece

% We'll compute "piece_after" and "king_after" predicates

% Pieces of Color after move (MColor, MType, MFX, MFY, MTX, MTY)
piece_after(MColor, MType, MTX, MTY, MFX, MFY, MTX, MTY) :-
    piece(MColor, MType, MFX, MFY).

piece_after(C, T, PX, PY, MFX, MFY, MTX, MTY) :-
    piece(C, T, PX, PY),
    not (PX = MFX, PY = MFY),
    not (PX = MTX, PY = MTY).

% Helper to avoid complex negation
same_sq(X1,Y1,X2,Y2) :- X1=X2, Y1=Y2.

piece_after2(C, T, PX, PY, MFX, MFY, MTX, MTY) :-
    piece(C, T, PX, PY),
    not same_sq(PX,PY,MFX,MFY),
    not same_sq(PX,PY,MTX,MTY).

% King after move
king_after(MColor, MTX, MTY, MFX, MFY, MTX, MTY) :-
    king(MColor, MFX, MFY).

king_after(C, KX, KY, MFX, MFY, MTX, MTY) :-
    king(C, KX, KY),
    not same_sq(KX,KY,MFX,MFY),
    not same_sq(KX,KY,MTX,MTY).

% occupied_after: any piece or king on square after move
occupied_after(X, Y, MFX, MFY, MTX, MTY) :-
    piece_after2(_, _, X, Y, MFX, MFY, MTX, MTY).
occupied_after(X, Y, MFX, MFY, MTX, MTY) :-
    piece_after2(MColor, _, MTX, MTY, MFX, MFY, MTX, MTY),
    X = MTX, Y = MTY,
    piece(MColor, _, MFX, MFY).
occupied_after(MTX, MTY, MFX, MFY, MTX, MTY) :-
    piece(_, _, MFX, MFY).
occupied_after(X, Y, MFX, MFY, MTX, MTY) :-
    king_after(_, X, Y, MFX, MFY, MTX, MTY).

% This is getting complex. Let me use a simpler approach:
% directly compute attacks on the black king's position after white's move.

% -------------------------------------------------------
% Simpler approach: compute attacks directly
% -------------------------------------------------------

% After white moves (FX,FY)->(TX,TY) with piece Type,
% does white attack the black king's square?

% Black king position (may not change since white is moving)
% Black king stays at (KX,KY) unless captured (which would end the game)

% We need to check if after the move, the white piece at (TX,TY) attacks (KX,KY)
% and the path is clear given the new board state.

% New board state after white move (WFX,WFY)->(WTX,WTY):
% - White piece removed from (WFX,WFY), placed at (WTX,WTY)
% - Any black piece at (WTX,WTY) is captured

% occupied_new(X,Y) after white move (WFX,WFY,WTX,WTY):
% = occupied(X,Y) except:
%   - (WFX,WFY) is now empty
%   - (WTX,WTY) has the white piece (may have captured black piece there)

% For check detection, we need to know if white attacks black king after the move.
% Black king is at (BKX,BKY).

% Let's define: after_move_white_attacks_bking(Type, WFX, WFY, WTX, WTY)

% Helper: is (BX,BY) strictly between (WTX,WTY) and (BKX,BKY) on a horizontal line,
% in the new board state?

% New occupied: piece at (X,Y) in new state
new_occ(X, Y, WFX, WFY, WTX, WTY) :-
    occupied(X, Y),
    not same_sq(X, Y, WFX, WFY),
    not same_sq(X, Y, WTX, WTY).
new_occ(WTX, WTY, WFX, WFY, WTX, WTY) :-
    piece(white, _, WFX, WFY).
new_occ(WTX, WTY, WFX, WFY, WTX, WTY) :-
    king(white, WFX, WFY).

% Blocking on horizontal after move
new_blocked_h(FX, FY, TX, WFX, WFY, WTX, WTY) :-
    new_occ(BX, FY, WFX, WFY, WTX, WTY),
    FX < TX, BX > FX, BX < TX.
new_blocked_h(FX, FY, TX, WFX, WFY, WTX, WTY) :-
    new_occ(BX, FY, WFX, WFY, WTX, WTY),
    FX > TX, BX < FX, BX > TX.

% Blocking on vertical after move
new_blocked_v(FX, FY, TY, WFX, WFY, WTX, WTY) :-
    new_occ(FX, BY, WFX, WFY, WTX, WTY),
    FY < TY, BY > FY, BY < TY.
new_blocked_v(FX, FY, TY, WFX, WFY, WTX, WTY) :-
    new_occ(FX, BY, WFX, WFY, WTX, WTY),
    FY > TY, BY < FY, BY > TY.

% Blocking on diagonal after move
new_blocked_d(FX, FY, TX, TY, WFX, WFY, WTX, WTY) :-
    new_occ(BX, BY, WFX, WFY, WTX, WTY),
    FX < TX, FY < TY, BX > FX, BX < TX, BY > FY, BY < TY,
    BX - FX = BY - FY.
new_blocked_d(FX, FY, TX, TY, WFX, WFY, WTX, WTY) :-
    new_occ(BX, BY, WFX, WFY, WTX, WTY),
    FX > TX, FY > TY, BX < FX, BX > TX, BY < FY, BY > TY,
    FX - BX = FY - BY.
new_blocked_d(FX, FY, TX, TY, WFX, WFY, WTX, WTY) :-
    new_occ(BX, BY, WFX, WFY, WTX, WTY),
    FX > TX, FY < TY, BX < FX, BX > TX, BY > FY, BY < TY,
    FX - BX = BY - FY.
new_blocked_d(FX, FY, TX, TY, WFX, WFY, WTX, WTY) :-
    new_occ(BX, BY, WFX, WFY, WTX, WTY),
    FX < TX, FY > TY, BX > FX, BX < TX, BY < FY, BY > TY,
    BX - FX = FY - BY.

% White queen at (WTX,WTY) attacks (BKX,BKY) in new state
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % horizontal
    WTY = BKY, WTX != BKX,
    not new_blocked_h(WTX, WTY, BKX, WFX, WFY, WTX, WTY).
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % vertical
    WTX = BKX, WTY != BKY,
    not new_blocked_v(WTX, WTY, BKY, WFX, WFY, WTX, WTY).
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % diagonal up-right
    WTX < BKX, WTY < BKY, BKX - WTX = BKY - WTY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % diagonal down-left
    WTX > BKX, WTY > BKY, WTX - BKX = WTY - BKY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % diagonal up-left
    WTX > BKX, WTY < BKY, WTX - BKX = BKY - WTY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    % diagonal down-right
    WTX < BKX, WTY > BKY, BKX - WTX = WTY - BKY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).

% White rook at (WTX,WTY) attacks (BKX,BKY)
white_rook_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTY = BKY, WTX != BKX,
    not new_blocked_h(WTX, WTY, BKX, WFX, WFY, WTX, WTY).
white_rook_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTX = BKX, WTY != BKY,
    not new_blocked_v(WTX, WTY, BKY, WFX, WFY, WTX, WTY).

% White bishop at (WTX,WTY) attacks (BKX,BKY)
white_bishop_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTX < BKX, WTY < BKY, BKX - WTX = BKY - WTY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_bishop_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTX > BKX, WTY > BKY, WTX - BKX = WTY - BKY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_bishop_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTX > BKX, WTY < BKY, WTX - BKX = BKY - WTY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
white_bishop_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY) :-
    WTX < BKX, WTY > BKY, BKX - WTX = WTY - BKY,
    not new_blocked_d(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).

% White knight at (WTX,WTY) attacks (BKX,BKY)
white_knight_attacks(WTX, WTY, BKX, BKY) :-
    knight_delta(DX, DY),
    BKX = WTX + DX, BKY = WTY + DY.

% White pawn at (WTX,WTY) attacks (BKX,BKY) (white moves up)
white_pawn_attacks(WTX, WTY, BKX, BKY) :-
    BKY = WTY + 1,
    (BKX = WTX + 1 ; BKX = WTX - 1).

% White king at (WTX,WTY) attacks (BKX,BKY)
white_king_attacks(WTX, WTY, BKX, BKY) :-
    king_delta(DX, DY),
    BKX = WTX + DX, BKY = WTY + DY.

% -------------------------------------------------------
% Does white piece at (WTX,WTY) of Type give check to black king at (BKX,BKY)?
% -------------------------------------------------------
gives_check(queen, WFX, WFY, WTX, WTY, BKX, BKY) :-
    white_queen_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
gives_check(rook, WFX, WFY, WTX, WTY, BKX, BKY) :-
    white_rook_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
gives_check(bishop, WFX, WFY, WTX, WTY, BKX, BKY) :-
    white_bishop_attacks(WTX, WTY, BKX, BKY, WFX, WFY, WTX, WTY).
gives_check(knight, WFX, WFY, WTX, WTY, BKX, BKY) :-
    white_knight_attacks(WTX, WTY, BKX, BKY).
gives_check(pawn, WFX, WFY, WTX, WTY, BKX, BKY) :-
    white_pawn_attacks(WTX, WTY, BKX, BKY).

% Also check for discovered check: other white pieces that now attack BK
% after the moving piece vacated (WFX,WFY)

% Other white pieces (not the one that moved) that attack BK in new state
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    % queen attacks BK horizontally
    QY = BKY, QX != BKX,
    not new_blocked_h(QX, QY, BKX, WFX, WFY, WTX, WTY).
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX = BKX, QY != BKY,
    not new_blocked_v(QX, QY, BKY, WFX, WFY, WTX, WTY).
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX < BKX, QY < BKY, BKX - QX = BKY - QY,
    not new_blocked_d(QX, QY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX > BKX, QY > BKY, QX - BKX = QY - BKY,
    not new_blocked_d(QX, QY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX > BKX, QY < BKY, QX - BKX = BKY - QY,
    not new_blocked_d(QX, QY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX < BKX, QY > BKY, BKX - QX = QY - BKY,
    not new_blocked_d(QX, QY, BKX, BKY, WFX, WFY, WTX, WTY).

other_white_rook_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, rook, RX, RY),
    not same_sq(RX, RY, WFX, WFY),
    RY = BKY, RX != BKX,
    not new_blocked_h(RX, RY, BKX, WFX, WFY, WTX, WTY).
other_white_rook_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, rook, RX, RY),
    not same_sq(RX, RY, WFX, WFY),
    RX = BKX, RY != BKY,
    not new_blocked_v(RX, RY, BKY, WFX, WFY, WTX, WTY).

other_white_bishop_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX < BKX, BY < BKY, BKX - BX = BKY - BY,
    not new_blocked_d(BX, BY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_bishop_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX > BKX, BY > BKY, BX - BKX = BY - BKY,
    not new_blocked_d(BX, BY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_bishop_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX > BKX, BY < BKY, BX - BKX = BKY - BY,
    not new_blocked_d(BX, BY, BKX, BKY, WFX, WFY, WTX, WTY).
other_white_bishop_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX < BKX, BY > BKY, BKX - BX = BY - BKY,
    not new_blocked_d(BX, BY, BKX, BKY, WFX, WFY, WTX, WTY).

% White king attacks BK (for discovered check via king move - unusual but possible)
other_white_king_attacks(BKX, BKY, WFX, WFY, WTX, WTY) :-
    king(white, KX, KY),
    not same_sq(KX, KY, WFX, WFY),
    king_delta(DX, DY),
    BKX = KX + DX, BKY = KY + DY.

% Combined: white attacks black king after move (WFX,WFY)->(WTX,WTY) with Type
white_attacks_bking_after(Type, WFX, WFY, WTX, WTY) :-
    king(black, BKX, BKY),
    gives_check(Type, WFX, WFY, WTX, WTY, BKX, BKY).

white_attacks_bking_after(Type, WFX, WFY, WTX, WTY) :-
    king(black, BKX, BKY),
    other_white_queen_attacks(BKX, BKY, WFX, WFY, WTX, WTY).

white_attacks_bking_after(Type, WFX, WFY, WTX, WTY) :-
    king(black, BKX, BKY),
    other_white_rook_attacks(BKX, BKY, WFX, WFY, WTX, WTY).

white_attacks_bking_after(Type, WFX, WFY, WTX, WTY) :-
    king(black, BKX, BKY),
    other_white_bishop_attacks(BKX, BKY, WFX, WFY, WTX, WTY).

% -------------------------------------------------------
% Check if black king can escape after white's move
% -------------------------------------------------------

% White attacks a square (TX,TY) in the new board state (after white move WFX,WFY->WTX,WTY)
% This is needed to determine if black king can move to (TX,TY)

% White queen (at WTX,WTY after move) attacks (TX,TY)
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTY = TY, WTX != TX,
    not new_blocked_h(WTX, WTY, TX, WFX, WFY, WTX, WTY).
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX = TX, WTY != TY,
    not new_blocked_v(WTX, WTY, TY, WFX, WFY, WTX, WTY).
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX < TX, WTY < TY, TX - WTX = TY - WTY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX > TX, WTY > TY, WTX - TX = WTY - TY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX > TX, WTY < TY, WTX - TX = TY - WTY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX < TX, WTY > TY, TX - WTX = WTY - TY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).

% White rook (at WTX,WTY) attacks (TX,TY)
wr_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTY = TY, WTX != TX,
    not new_blocked_h(WTX, WTY, TX, WFX, WFY, WTX, WTY).
wr_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX = TX, WTY != TY,
    not new_blocked_v(WTX, WTY, TY, WFX, WFY, WTX, WTY).

% White bishop (at WTX,WTY) attacks (TX,TY)
wb_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX < TX, WTY < TY, TX - WTX = TY - WTY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wb_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX > TX, WTY > TY, WTX - TX = WTY - TY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wb_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX > TX, WTY < TY, WTX - TX = TY - WTY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
wb_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY) :-
    WTX < TX, WTY > TY, TX - WTX = WTY - TY,
    not new_blocked_d(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).

% White knight (at WTX,WTY) attacks (TX,TY)
wn_attacks_sq(WTX, WTY, TX, TY) :-
    knight_delta(DX, DY),
    TX = WTX + DX, TY = WTY + DY.

% White pawn (at WTX,WTY) attacks (TX,TY)
wp_attacks_sq(WTX, WTY, TX, TY) :-
    TY = WTY + 1,
    (TX = WTX + 1 ; TX = WTX - 1).

% White king (at WTX,WTY) attacks (TX,TY)
wk_attacks_sq(WTX, WTY, TX, TY) :-
    king_delta(DX, DY),
    TX = WTX + DX, TY = WTY + DY.

% Other white queens (not the moved piece) attack (TX,TY) in new state
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QY = TY, QX != TX,
    not new_blocked_h(QX, QY, TX, WFX, WFY, WTX, WTY).
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX = TX, QY != TY,
    not new_blocked_v(QX, QY, TY, WFX, WFY, WTX, WTY).
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX < TX, QY < TY, TX - QX = TY - QY,
    not new_blocked_d(QX, QY, TX, TY, WFX, WFY, WTX, WTY).
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX > TX, QY > TY, QX - TX = QY - TY,
    not new_blocked_d(QX, QY, TX, TY, WFX, WFY, WTX, WTY).
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX > TX, QY < TY, QX - TX = TY - QY,
    not new_blocked_d(QX, QY, TX, TY, WFX, WFY, WTX, WTY).
owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, queen, QX, QY),
    not same_sq(QX, QY, WFX, WFY),
    QX < TX, QY > TY, TX - QX = QY - TY,
    not new_blocked_d(QX, QY, TX, TY, WFX, WFY, WTX, WTY).

owr_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, rook, RX, RY),
    not same_sq(RX, RY, WFX, WFY),
    RY = TY, RX != TX,
    not new_blocked_h(RX, RY, TX, WFX, WFY, WTX, WTY).
owr_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, rook, RX, RY),
    not same_sq(RX, RY, WFX, WFY),
    RX = TX, RY != TY,
    not new_blocked_v(RX, RY, TY, WFX, WFY, WTX, WTY).

owb_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX < TX, BY < TY, TX - BX = TY - BY,
    not new_blocked_d(BX, BY, TX, TY, WFX, WFY, WTX, WTY).
owb_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX > TX, BY > TY, BX - TX = BY - TY,
    not new_blocked_d(BX, BY, TX, TY, WFX, WFY, WTX, WTY).
owb_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX > TX, BY < TY, BX - TX = TY - BY,
    not new_blocked_d(BX, BY, TX, TY, WFX, WFY, WTX, WTY).
owb_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, BX, BY),
    not same_sq(BX, BY, WFX, WFY),
    BX < TX, BY > TY, TX - BX = BY - TY,
    not new_blocked_d(BX, BY, TX, TY, WFX, WFY, WTX, WTY).

own_attacks_sq(TX, TY, WFX, WFY) :-
    piece(white, knight, NX, NY),
    not same_sq(NX, NY, WFX, WFY),
    knight_delta(DX, DY),
    TX = NX + DX, TY = NY + DY.

owp_attacks_sq(TX, TY, WFX, WFY) :-
    piece(white, pawn, PX, PY),
    not same_sq(PX, PY, WFX, WFY),
    TY = PY + 1,
    (TX = PX + 1 ; TX = PX - 1).

owk_attacks_sq(TX, TY, WFX, WFY, WTX, WTY) :-
    king(white, KX, KY),
    not same_sq(KX, KY, WFX, WFY),
    king_delta(DX, DY),
    TX = KX + DX, TY = KY + DY.

% White king (moved to WTX,WTY) attacks (TX,TY)
moved_wk_attacks_sq(WTX, WTY, TX, TY) :-
    king_delta(DX, DY),
    TX = WTX + DX, TY = WTY + DY.

% Combined: white attacks square (TX,TY) after move (WFX,WFY,WTX,WTY) with Type
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    piece(white, queen, WFX, WFY),
    Type = queen,
    wq_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    piece(white, rook, WFX, WFY),
    Type = rook,
    wr_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    piece(white, bishop, WFX, WFY),
    Type = bishop,
    wb_attacks_sq(WTX, WTY, TX, TY, WFX, WFY, WTX, WTY).
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    piece(white, knight, WFX, WFY),
    Type = knight,
    wn_attacks_sq(WTX, WTY, TX, TY).
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    piece(white, pawn, WFX, WFY),
    Type = pawn,
    wp_attacks_sq(WTX, WTY, TX, TY).
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    king(white, WFX, WFY),
    Type = king,
    moved_wk_attacks_sq(WTX, WTY, TX, TY).

% Other white pieces attack (TX,TY) after move
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    owq_attacks_sq(TX, TY, WFX, WFY, WTX, WTY),
    Type = queen.
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    owr_attacks_sq(TX, TY, WFX, WFY, WTX, WTY),
    Type = rook.
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    owb_attacks_sq(TX, TY, WFX, WFY, WTX, WTY),
    Type = bishop.
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    own_attacks_sq(TX, TY, WFX, WFY),
    Type = knight.
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    owp_attacks_sq(TX, TY, WFX, WFY),
    Type = pawn.
white_attacks_sq_after(TX, TY, Type, WFX, WFY, WTX, WTY) :-
    owk_attacks_sq(TX, TY, WFX, WFY, WTX, WTY),
    Type = king.

% -------------------------------------------------------
% Black king escape squares after white move
% -------------------------------------------------------

% Black king can potentially move to (EX,EY)
bk_escape_candidate(EX, EY) :-
    king(black, BKX, BKY),
    king_delta(DX, DY),
    EX = BKX + DX, EY = BKY + DY,
    file(EX), rank(EY).

% Black piece (non-king) at (X,Y) - blocks escape
black_piece_at(X, Y) :- piece(black, _, X, Y).

% After white move (WFX,WFY)->(WTX,WTY), black king can escape to (EX,EY) if:
% 1. (EX,EY) is a valid candidate
% 2. Not occupied by black piece (own piece)
% 3. Not attacked by white after the move
% 4. (EX,EY) != (WTX,WTY) unless it was a capture (black piece there)
%    Actually if white moved to (WTX,WTY), that square is now white-occupied

% Black king cannot move to a square occupied by its own pieces
% or to a square attacked by white, or to the square white just moved to (white piece there)

bk_can_escape(EX, EY, Type, WFX, WFY, WTX, WTY) :-
    bk_escape_candidate(EX, EY),
    not black_piece_at(EX, EY),
    not same_sq(EX, EY, WTX, WTY),  % white piece now there (unless captured - but white piece is there)
    not white_attacks_sq_after(EX, EY, _, WFX, WFY, WTX, WTY).

% Wait - if black king captures the white piece at (WTX,WTY), that's a valid escape
% if the resulting square is not attacked by other white pieces
bk_can_escape_capture(WTX, WTY, Type, WFX, WFY, WTX, WTY) :-
    bk_escape_candidate(WTX, WTY),
    not black_piece_at(WTX, WTY),
    % After capture, the moved white piece is gone, so we need to check
    % if other white pieces attack (WTX,WTY)
    not owq_attacks_sq(WTX, WTY, WFX, WFY, WTX, WTY),
    not owr_attacks_sq(WTX, WTY, WFX, WFY, WTX, WTY),
    not owb_attacks_sq(WTX, WTY, WFX, WFY, WTX, WTY),
    not own_attacks_sq(WTX, WTY, WFX, WFY),
    not owp_attacks_sq(WTX, WTY, WFX, WFY),
    not owk_attacks_sq(WTX, WTY, WFX, WFY, WTX, WTY).

% Black king has an escape if either bk_can_escape or bk_can_escape_capture
bk_has_escape(Type, WFX, WFY, WTX, WTY) :-
    bk_can_escape(_, _, Type, WFX, WFY, WTX, WTY).
bk_has_escape(Type, WFX, WFY, WTX, WTY) :-
    bk_can_escape_capture(WTX, WTY, Type, WFX, WFY, WTX, WTY).

% -------------------------------------------------------
% Candidate white moves
% -------------------------------------------------------
white_move(queen, FX, FY, TX, TY) :-
    to_move(white),
    queen_can_move(white, FX, FY, TX, TY).
white_move(rook, FX, FY, TX, TY) :-
    to_move(white),
    rook_can_move(white, FX, FY, TX, TY).
white_move(bishop, FX, FY, TX, TY) :-
    to_move(white),
    bishop_can_move(white, FX, FY, TX, TY).
white_move(knight, FX, FY, TX, TY) :-
    to_move(white),
    knight_can_move(white, FX, FY, TX, TY).
white_move(pawn, FX, FY, TX, TY) :-
    to_move(white),
    pawn_can_move(white, FX, FY, TX, TY).
white_move(king, FX, FY, TX, TY) :-
    to_move(white),
    king_can_move(white, FX, FY, TX, TY).

% -------------------------------------------------------
% Mate in one: white move that gives check and black has no escape
% -------------------------------------------------------
mate_move(white, Type, FX, FY, TX, TY) :-
    to_move(white),
    white_move(Type, FX, FY, TX, TY),
    white_attacks_bking_after(Type, FX, FY, TX, TY),
    not bk_has_escape(Type, FX, FY, TX, TY).

#show mate_move/6.