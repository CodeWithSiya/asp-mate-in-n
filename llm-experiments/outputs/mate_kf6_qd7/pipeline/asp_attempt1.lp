% Domain definitions
file(1..8).
rank(1..8).

% Opponent color
opponent(white, black).
opponent(black, white).

% All occupied squares (by any piece)
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

% White king position helper
white_king(KX, KY) :- king(white, KX, KY).
black_king(KX, KY) :- king(black, KX, KY).

%-----------------------------------------------------------------------
% Queen move generation from (FX,FY) to (TX,TY)
%-----------------------------------------------------------------------

% Horizontal queen move (same rank)
queen_move_h(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TY = FY,
    TX != FX.

% Vertical queen move (same file)
queen_move_v(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX = FX,
    TY != FY.

% Diagonal queen move
queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX,
    DX = TX - FX, DX >= 0,
    DY = TY - FY, DY >= 0,
    DX = DY.

queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX,
    DX = FX - TX, DX > 0,
    DY = TY - FY, DY >= 0,
    DX = DY.

queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX,
    DX = TX - FX, DX >= 0,
    DY = FY - TY, DY > 0,
    DX = DY.

queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX != FX,
    DX = FX - TX, DX > 0,
    DY = FY - TY, DY > 0,
    DX = DY.

%-----------------------------------------------------------------------
% Path blocking for queen moves
%-----------------------------------------------------------------------

% Horizontal blocking: something between FX and TX on rank FY
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    FX > TX, BX < FX, BX > TX.

% Vertical blocking: something between FY and TY on file FX
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    FY > TY, BY < FY, BY > TY.

% Diagonal blocking: something strictly between (FX,FY) and (TX,TY) on diagonal
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    FX < TX, FY < TY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.

blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    FX > TX, FY < TY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.

blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    FX < TX, FY > TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.

blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    FX > TX, FY > TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

%-----------------------------------------------------------------------
% Valid queen moves (not blocked, not landing on own piece)
%-----------------------------------------------------------------------

% Own pieces (white)
own_piece(X, Y) :- piece(white, _, X, Y).
own_piece(X, Y) :- king(white, X, Y).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_h(FX, FY, TX, TY),
    not blocked_h(FX, FY, TX),
    not own_piece(TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_v(FX, FY, TX, TY),
    not blocked_v(FX, FY, TY),
    not own_piece(TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d(FX, FY, TX, TY),
    not blocked_d(FX, FY, TX, TY),
    not own_piece(TX, TY).

%-----------------------------------------------------------------------
% After queen moves to (TX,TY), does it attack square (AX,AY)?
%-----------------------------------------------------------------------

% Horizontal attack from (TX,TY) to (AX,AY)
queen_attacks_h(TX, TY, AX, AY) :-
    AY = TY, AX != TX,
    file(AX), rank(AY).

% Vertical attack
queen_attacks_v(TX, TY, AX, AY) :-
    AX = TX, AY != TY,
    file(AX), rank(AY).

% Diagonal attack
queen_attacks_d(TX, TY, AX, AY) :-
    AX != TX,
    DX = AX - TX, DX >= 0,
    DY = AY - TY, DY >= 0,
    DX = DY,
    file(AX), rank(AY).

queen_attacks_d(TX, TY, AX, AY) :-
    AX != TX,
    DX = TX - AX, DX > 0,
    DY = AY - TY, DY >= 0,
    DX = DY,
    file(AX), rank(AY).

queen_attacks_d(TX, TY, AX, AY) :-
    AX != TX,
    DX = AX - TX, DX >= 0,
    DY = TY - AY, DY > 0,
    DX = DY,
    file(AX), rank(AY).

queen_attacks_d(TX, TY, AX, AY) :-
    AX != TX,
    DX = TX - AX, DX > 0,
    DY = TY - AY, DY > 0,
    DX = DY,
    file(AX), rank(AY).

%-----------------------------------------------------------------------
% After queen moves from (FX,FY) to (TX,TY), blocking for attacks
% The board changes: queen is no longer at (FX,FY), may capture at (TX,TY)
%-----------------------------------------------------------------------

% Occupied squares after queen moves from (FX,FY) to (TX,TY)
occ_after(BX, BY, FX, FY, TX, TY) :-
    occupied(BX, BY),
    not (BX = FX, BY = FY),   % queen left
    not (BX = TX, BY = TY).   % captured piece removed

occ_after(TX, TY, FX, FY, TX, TY) :-  % queen now at (TX,TY)
    queen_can_move(FX, FY, TX, TY).

% Blocking after move for horizontal queen attack from (TX,TY) to (AX,AY)
blocked_h_after(TX, TY, AX, FX, FY, QTX, QTY) :-
    occ_after(BX, TY, FX, FY, QTX, QTY),
    TX < AX, BX > TX, BX < AX.
blocked_h_after(TX, TY, AX, FX, FY, QTX, QTY) :-
    occ_after(BX, TY, FX, FY, QTX, QTY),
    TX > AX, BX < TX, BX > AX.

blocked_v_after(TX, TY, AY, FX, FY, QTX, QTY) :-
    occ_after(TX, BY, FX, FY, QTX, QTY),
    TY < AY, BY > TY, BY < AY.
blocked_v_after(TX, TY, AY, FX, FY, QTX, QTY) :-
    occ_after(TX, BY, FX, FY, QTX, QTY),
    TY > AY, BY < TY, BY > AY.

blocked_d_after(TX, TY, AX, AY, FX, FY, QTX, QTY) :-
    occ_after(BX, BY, FX, FY, QTX, QTY),
    TX < AX, TY < AY,
    BX > TX, BX < AX,
    BY > TY, BY < AY,
    BX - TX = BY - TY.

blocked_d_after(TX, TY, AX, AY, FX, FY, QTX, QTY) :-
    occ_after(BX, BY, FX, FY, QTX, QTY),
    TX > AX, TY < AY,
    BX < TX, BX > AX,
    BY > TY, BY < AY,
    TX - BX = BY - TY.

blocked_d_after(TX, TY, AX, AY, FX, FY, QTX, QTY) :-
    occ_after(BX, BY, FX, FY, QTX, QTY),
    TX < AX, TY > AY,
    BX > TX, BX < AX,
    BY < TY, BY > AY,
    BX - TX = TY - BY.

blocked_d_after(TX, TY, AX, AY, FX, FY, QTX, QTY) :-
    occ_after(BX, BY, FX, FY, QTX, QTY),
    TX > AX, TY > AY,
    BX < TX, BX > AX,
    BY < TY, BY > AY,
    TX - BX = TY - BY.

%-----------------------------------------------------------------------
% Queen (now at QTX,QTY) attacks square (AX,AY) after move from (FX,FY)
%-----------------------------------------------------------------------

queen_attacks_after(QTX, QTY, AX, AY, FX, FY) :-
    queen_attacks_h(QTX, QTY, AX, AY),
    not blocked_h_after(QTX, QTY, AX, FX, FY, QTX, QTY).

queen_attacks_after(QTX, QTY, AX, AY, FX, FY) :-
    queen_attacks_v(QTX, QTY, AX, AY),
    not blocked_v_after(QTX, QTY, AY, FX, FY, QTX, QTY).

queen_attacks_after(QTX, QTY, AX, AY, FX, FY) :-
    queen_attacks_d(QTX, QTY, AX, AY),
    not blocked_d_after(QTX, QTY, AX, AY, FX, FY, QTX, QTY).

%-----------------------------------------------------------------------
% White king attacks
%-----------------------------------------------------------------------
king_attacks(KX, KY, AX, AY) :-
    king(white, KX, KY),
    file(AX), rank(AY),
    DX = AX - KX, DX >= 0, DX <= 1,
    DY = AY - KY, DY >= 0, DY <= 1,
    not (AX = KX, AY = KY).

king_attacks(KX, KY, AX, AY) :-
    king(white, KX, KY),
    file(AX), rank(AY),
    DX = KX - AX, DX > 0, DX <= 1,
    DY = AY - KY, DY >= 0, DY <= 1,
    not (AX = KX, AY = KY).

king_attacks(KX, KY, AX, AY) :-
    king(white, KX, KY),
    file(AX), rank(AY),
    DX = AX - KX, DX >= 0, DX <= 1,
    DY = KY - AY, DY > 0, DY <= 1,
    not (AX = KX, AY = KY).

king_attacks(KX, KY, AX, AY) :-
    king(white, KX, KY),
    file(AX), rank(AY),
    DX = KX - AX, DX > 0, DX <= 1,
    DY = KY - AY, DY > 0, DY <= 1,
    not (AX = KX, AY = KY).

%-----------------------------------------------------------------------
% Square attacked by white after queen moves from (FX,FY) to (QTX,QTY)
%-----------------------------------------------------------------------
white_attacks_after(AX, AY, FX, FY, QTX, QTY) :-
    queen_attacks_after(QTX, QTY, AX, AY, FX, FY).

white_attacks_after(AX, AY, FX, FY, QTX, QTY) :-
    king_attacks(KX, KY, AX, AY),
    % king position unchanged
    file(AX), rank(AY).

%-----------------------------------------------------------------------
% Black king escape squares after queen moves from (FX,FY) to (QTX,QTY)
%-----------------------------------------------------------------------

% Adjacent squares to black king
adj_to_black_king(AX, AY) :-
    black_king(KX, KY),
    file(AX), rank(AY),
    DX = AX - KX, DX >= 0, DX <= 1,
    DY = AY - KY, DY >= 0, DY <= 1,
    not (AX = KX, AY = KY).

adj_to_black_king(AX, AY) :-
    black_king(KX, KY),
    file(AX), rank(AY),
    DX = KX - AX, DX > 0, DX <= 1,
    DY = AY - KY, DY >= 0, DY <= 1,
    not (AX = KX, AY = KY).

adj_to_black_king(AX, AY) :-
    black_king(KX, KY),
    file(AX), rank(AY),
    DX = AX - KX, DX >= 0, DX <= 1,
    DY = KY - AY, DY > 0, DY <= 1,
    not (AX = KX, AY = KY).

adj_to_black_king(AX, AY) :-
    black_king(KX, KY),
    file(AX), rank(AY),
    DX = KX - AX, DX > 0, DX <= 1,
    DY = KY - AY, DY > 0, DY <= 1,
    not (AX = KX, AY = KY).

% White piece at square (after move)
white_piece_after(X, Y, FX, FY, QTX, QTY) :-
    piece(white, _, X, Y),
    not (X = FX, Y = FY).
white_piece_after(QTX, QTY, FX, FY, QTX, QTY) :-
    queen_can_move(FX, FY, QTX, QTY).
white_piece_after(X, Y, FX, FY, QTX, QTY) :-
    king(white, X, Y).

% Black king can escape to (AX,AY) if:
% 1. It's adjacent to black king
% 2. Not occupied by a black piece (only black piece is the king itself)
% 3. Not attacked by white after the move
% 4. Not occupied by a white piece after the move (king can capture if not defended, but we handle via attacks)

% Actually: black king can move to AX,AY if not white_piece_after OR if it's the queen (capturable if not defended)
% Simplification: black king can't move to square with white piece that is defended by white
% Black king can capture white queen if queen is not defended

% Queen defended after move: white king attacks queen position
queen_defended(QTX, QTY) :-
    king_attacks(KX, KY, QTX, QTY).

% Black king can go to (AX,AY) after white queen moves (FX,FY)->(QTX,QTY):
% - adjacent to black king
% - not attacked by white (queen or king) after move
% - if white piece is there: can only go if it's the queen and queen is not defended
%   but if queen is there and defended, it's attacked so covered by white_attacks_after

% Escape: adjacent, not attacked by white after move, not blocked by own (black) pieces
% Black has no other pieces here, so just:
can_escape(AX, AY, FX, FY, QTX, QTY) :-
    adj_to_black_king(AX, AY),
    not white_attacks_after(AX, AY, FX, FY, QTX, QTY),
    not white_piece_after(AX, AY, FX, FY, QTX, QTY).

% Also black king can capture white queen if queen is not defended
can_escape(QTX, QTY, FX, FY, QTX, QTY) :-
    adj_to_black_king(QTX, QTY),
    queen_can_move(FX, FY, QTX, QTY),
    not queen_defended(QTX, QTY).

%-----------------------------------------------------------------------
% Black king in check after queen moves to (QTX,QTY)
%-----------------------------------------------------------------------
in_check_after(FX, FY, QTX, QTY) :-
    black_king(KX, KY),
    white_attacks_after(KX, KY, FX, FY, QTX, QTY).

%-----------------------------------------------------------------------
% Checkmate: black king in check and no escape
%-----------------------------------------------------------------------
is_checkmate(FX, FY, QTX, QTY) :-
    queen_can_move(FX, FY, QTX, QTY),
    in_check_after(FX, FY, QTX, QTY),
    not can_escape(_, _, FX, FY, QTX, QTY).

%-----------------------------------------------------------------------
% Output
%-----------------------------------------------------------------------
mate_move(white, queen, FX, FY, QTX, QTY) :-
    is_checkmate(FX, FY, QTX, QTY).

#show mate_move/6.