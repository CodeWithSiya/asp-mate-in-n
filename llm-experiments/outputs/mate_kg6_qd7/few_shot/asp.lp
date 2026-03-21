% Board coordinates
coord(1..8).
square(X,Y) :- coord(X), coord(Y).

% Side to move
to_move(white).

% Pieces
piece(black, king, 8, 8).
king(black, 8, 8).
piece(white, queen, 4, 7).
piece(white, king, 7, 6).
king(white, 7, 6).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper: occupied squares
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
occupied(X,Y) :- piece(_,_,X,Y).
occupied_by(Color,X,Y) :- piece(Color,_,X,Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Queen moves: rook-like (horizontal/vertical) and bishop-like (diagonal)
% We enumerate directions and sliding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Directions for queen (and rook part)
rook_dir(1,0). rook_dir(-1,0). rook_dir(0,1). rook_dir(0,-1).
bishop_dir(1,1). bishop_dir(1,-1). bishop_dir(-1,1). bishop_dir(-1,-1).
queen_dir(DX,DY) :- rook_dir(DX,DY).
queen_dir(DX,DY) :- bishop_dir(DX,DY).

% Queen sliding: reachable squares (no piece blocking)
% queen_slide(FX,FY,DX,DY,TX,TY) means queen at FX,FY can reach TX,TY in direction DX,DY
% Step 1
queen_slide(FX,FY,DX,DY,FX+DX,FY+DY) :-
    piece(white,queen,FX,FY),
    queen_dir(DX,DY),
    square(FX+DX,FY+DY),
    not occupied(FX+DX,FY+DY).

queen_slide_capture(FX,FY,DX,DY,FX+DX,FY+DY) :-
    piece(white,queen,FX,FY),
    queen_dir(DX,DY),
    square(FX+DX,FY+DY),
    occupied_by(black,FX+DX,FY+DY).

% Continue sliding if previous square was empty
queen_slide(FX,FY,DX,DY,TX+DX,TY+DY) :-
    queen_slide(FX,FY,DX,DY,TX,TY),
    square(TX+DX,TY+DY),
    not occupied(TX+DX,TY+DY).

queen_slide_capture(FX,FY,DX,DY,TX+DX,TY+DY) :-
    queen_slide(FX,FY,DX,DY,TX,TY),
    square(TX+DX,TY+DY),
    occupied_by(black,TX+DX,TY+DY).

% Queen can move to empty square
queen_can_move(FX,FY,TX,TY) :- queen_slide(FX,FY,_,_,TX,TY).
% Queen can capture
queen_can_move(FX,FY,TX,TY) :- queen_slide_capture(FX,FY,_,_,TX,TY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% King moves (white king)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
king_delta(-1,-1). king_delta(-1,0). king_delta(-1,1).
king_delta(0,-1).                    king_delta(0,1).
king_delta(1,-1).  king_delta(1,0).  king_delta(1,1).

white_king_can_move(FX,FY,TX,TY) :-
    king(white,FX,FY),
    king_delta(DX,DY),
    TX = FX+DX, TY = FY+DY,
    square(TX,TY),
    not occupied_by(white,TX,TY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate white's candidate moves
% move_white(piece_type, FX,FY,TX,TY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
move_white(queen,FX,FY,TX,TY) :-
    piece(white,queen,FX,FY),
    queen_can_move(FX,FY,TX,TY).

move_white(king,FX,FY,TX,TY) :-
    white_king_can_move(FX,FY,TX,TY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% After white makes a move, compute resulting board state
% We track: where is white queen, white king, black king
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% After white queen moves from (QFX,QFY) to (QTX,QTY):
% white queen new pos
wq_after(QTX,QTY,QTX,QTY,WKX,WKY,BKX,BKY) :-
    move_white(queen,QFX,QFY,QTX,QTY),
    king(white,WKX,WKY),
    king(black,BKX,BKY),
    % black king not captured (queen doesn't capture king in legal play, but let's keep)
    not (QTX = BKX, QTY = BKY). % queen can't capture king (illegal)

% Actually let's simplify: represent the resulting position after white's move
% result_pos(MoveID, WQX,WQY, WKX,WKY, BKX,BKY)
% where MoveID encodes the move

% We'll use the target square as move identifier for queen moves
% and separately for king moves

% After queen move to (TX,TY): queen is at (TX,TY), white king unchanged, black king unchanged (unless captured - but we skip king captures)
after_queen_move(TX,TY, TX,TY, WKX,WKY, BKX,BKY) :-
    move_white(queen,_,_,TX,TY),
    king(white,WKX,WKY),
    king(black,BKX,BKY),
    not (TX = BKX, TY = BKY).

% After white king move to (TX,TY): king is at (TX,TY), queen unchanged (unless king captures queen - not possible here since queen is white)
after_king_move(TX,TY, QX,QY, TX,TY, BKX,BKY) :-
    move_white(king,_,_,TX,TY),
    piece(white,queen,QX,QY),
    king(black,BKX,BKY),
    not (TX = BKX, TY = BKY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check detection: does white attack the black king square?
% After a queen move to (QX,QY), white king at (WKX,WKY), black king at (BKX,BKY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Queen attacks a square if it can slide to it (no blocking pieces)
% After the move, the board has: white queen at QX,QY; white king at WKX,WKY; black king at BKX,BKY
% (original queen square is now empty)

% We need to check if queen at (QX,QY) attacks (BKX,BKY)
% with only white king at (WKX,WKY) and black king at (BKX,BKY) on board

% Queen attacks along a direction if path is clear
% Helper: queen_attacks_sq(QX,QY, WKX,WKY, BKX,BKY) - queen at QX,QY attacks BKX,BKY
% given white king at WKX,WKY blocks

% Check if queen at (QX,QY) attacks (TX,TY) with blocker at (BLX,BLY)
% Direction from queen to target
% We need: same row, col, or diagonal, and no piece between them

% Same row: QY = TY, path clear
% Same col: QX = TX, path clear  
% Same diagonal: |QX-TX| = |QY-TY|, path clear

% Let's compute direction from queen to black king
% and check if path is clear (only blocker is white king)

% queen_attacks_king_after_qmove(QTX,QTY): after queen moves to QTX,QTY, it attacks black king
queen_attacks_bk(QX,QY,WKX,WKY,BKX,BKY) :-
    after_queen_move(QX,QY,QX,QY,WKX,WKY,BKX,BKY),
    % same row
    QY = BKY,
    % check no blocker between them on row
    not blocker_on_row(QX,QY,BKX,BKY,WKX,WKY).

queen_attacks_bk(QX,QY,WKX,WKY,BKX,BKY) :-
    after_queen_move(QX,QY,QX,QY,WKX,WKY,BKX,BKY),
    % same col
    QX = BKX,
    not blocker_on_col(QX,QY,BKX,BKY,WKX,WKY).

queen_attacks_bk(QX,QY,WKX,WKY,BKX,BKY) :-
    after_queen_move(QX,QY,QX,QY,WKX,WKY,BKX,BKY),
    % same diagonal
    DX = BKX - QX, DY = BKY - QY,
    DX = DY,
    not blocker_on_diag(QX,QY,BKX,BKY,WKX,WKY).

queen_attacks_bk(QX,QY,WKX,WKY,BKX,BKY) :-
    after_queen_move(QX,QY,QX,QY,WKX,WKY,BKX,BKY),
    % anti-diagonal
    DX = BKX - QX, DY = BKY - QY,
    DX = -DY,
    DX != 0,
    not blocker_on_antidiag(QX,QY,BKX,BKY,WKX,WKY).

% Blocker on row between QX,QY and BKX,BKY (same row QY=BKY)
% White king at WKX,WKY is a blocker if WKY = QY and WKX is strictly between QX and BKX
blocker_on_row(QX,QY,BKX,BKY,WKX,WKY) :-
    WKY = QY, WKY = BKY,
    WKX > QX, WKX < BKX.
blocker_on_row(QX,QY,BKX,BKY,WKX,WKY) :-
    WKY = QY, WKY = BKY,
    WKX < QX, WKX > BKX.

blocker_on_col(QX,QY,BKX,BKY,WKX,WKY) :-
    WKX = QX, WKX = BKX,
    WKY > QY, WKY < BKY.
blocker_on_col(QX,QY,BKX,BKY,WKX,WKY) :-
    WKX = QX, WKX = BKX,
    WKY < QY, WKY > BKY.

% Diagonal blocker: WK is on same diagonal between Q and BK
blocker_on_diag(QX,QY,BKX,BKY,WKX,WKY) :-
    % same diagonal as Q->BK: DX=DY
    DX = BKX - QX, DY = BKY - QY, DX = DY,
    % WK on same diagonal
    WDX = WKX - QX, WDY = WKY - QY, WDX = WDY,
    % WK strictly between Q and BK
    WDX != 0,
    WDX > 0, DX > 0, WDX < DX.
blocker_on_diag(QX,QY,BKX,BKY,WKX,WKY) :-
    DX = BKX - QX, DY = BKY - QY, DX = DY,
    WDX = WKX - QX, WDY = WKY - QY, WDX = WDY,
    WDX != 0,
    WDX < 0, DX < 0, WDX > DX.

blocker_on_antidiag(QX,QY,BKX,BKY,WKX,WKY) :-
    DX = BKX - QX, DY = BKY - QY, DX = -DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = -WDY, WDX != 0,
    WDX > 0, DX > 0, WDX < DX.
blocker_on_antidiag(QX,QY,BKX,BKY,WKX,WKY) :-
    DX = BKX - QX, DY = BKY - QY, DX = -DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = -WDY, WDX != 0,
    WDX < 0, DX < 0, WDX > DX.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% White king attacks a square
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wking_attacks(WKX,WKY,TX,TY) :-
    king(white,WKX,WKY),
    king_delta(DX,DY),
    TX = WKX+DX, TY = WKY+DY,
    square(TX,TY).

% After queen move, does white king attack black king?
wking_attacks_bk_after_qmove(QTX,QTY) :-
    after_queen_move(QTX,QTY,_,_,WKX,WKY,BKX,BKY),
    wking_attacks(WKX,WKY,BKX,BKY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Is black king in check after white queen moves to (QTX,QTY)?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bk_in_check_after_qmove(QTX,QTY) :-
    queen_attacks_bk(QTX,QTY,WKX,WKY,BKX,BKY),
    after_queen_move(QTX,QTY,QTX,QTY,WKX,WKY,BKX,BKY).

bk_in_check_after_qmove(QTX,QTY) :-
    wking_attacks_bk_after_qmove(QTX,QTY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Black king's possible escape moves after white queen moved to (QTX,QTY)
% Black king at (BKX,BKY), white king at (WKX,WKY), white queen at (QTX,QTY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Black king can move to (TX,TY) if:
% 1. Square is on board
% 2. Not occupied by black piece (none here)
% 3. Not attacked by white queen at (QTX,QTY)
% 4. Not attacked by white king at (WKX,WKY)
% 5. Not the same as white queen square (capture is ok if not defended)
%    Actually black king can capture white queen if that square is not defended by white king

% Square attacked by white queen at (QX,QY) with white king at (WKX,WKY)
% (black king has moved away, so we need to check attacks on candidate square)

% queen_attacks_sq_general(QX,QY,WKX,WKY,TX,TY): queen at QX,QY attacks TX,TY, white king at WKX,WKY
queen_attacks_sq(QX,QY,WKX,WKY,TX,TY) :-
    coord(QX), coord(QY), coord(WKX), coord(WKY), coord(TX), coord(TY),
    QY = TY, QX != TX,
    not blocker_on_row_gen(QX,QY,TX,TY,WKX,WKY).

queen_attacks_sq(QX,QY,WKX,WKY,TX,TY) :-
    coord(QX), coord(QY), coord(WKX), coord(WKY), coord(TX), coord(TY),
    QX = TX, QY != TY,
    not blocker_on_col_gen(QX,QY,TX,TY,WKX,WKY).

queen_attacks_sq(QX,QY,WKX,WKY,TX,TY) :-
    coord(QX), coord(QY), coord(WKX), coord(WKY), coord(TX), coord(TY),
    DX = TX - QX, DY = TY - QY, DX = DY, DX != 0,
    not blocker_on_diag_gen(QX,QY,TX,TY,WKX,WKY).

queen_attacks_sq(QX,QY,WKX,WKY,TX,TY) :-
    coord(QX), coord(QY), coord(WKX), coord(WKY), coord(TX), coord(TY),
    DX = TX - QX, DY = TY - QY, DX = -DY, DX != 0,
    not blocker_on_antidiag_gen(QX,QY,TX,TY,WKX,WKY).

% Blocker helpers (general)
blocker_on_row_gen(QX,QY,TX,TY,WKX,WKY) :-
    WKY = QY, WKY = TY,
    WKX > QX, WKX < TX.
blocker_on_row_gen(QX,QY,TX,TY,WKX,WKY) :-
    WKY = QY, WKY = TY,
    WKX < QX, WKX > TX.

blocker_on_col_gen(QX,QY,TX,TY,WKX,WKY) :-
    WKX = QX, WKX = TX,
    WKY > QY, WKY < TY.
blocker_on_col_gen(QX,QY,TX,TY,WKX,WKY) :-
    WKX = QX, WKX = TX,
    WKY < QY, WKY > TY.

blocker_on_diag_gen(QX,QY,TX,TY,WKX,WKY) :-
    DX = TX - QX, DY = TY - QY, DX = DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = WDY, WDX != 0,
    WDX > 0, DX > 0, WDX < DX.
blocker_on_diag_gen(QX,QY,TX,TY,WKX,WKY) :-
    DX = TX - QX, DY = TY - QY, DX = DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = WDY, WDX != 0,
    WDX < 0, DX < 0, WDX > DX.

blocker_on_antidiag_gen(QX,QY,TX,TY,WKX,WKY) :-
    DX = TX - QX, DY = TY - QY, DX = -DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = -WDY, WDX != 0,
    WDX > 0, DX > 0, WDX < DX.
blocker_on_antidiag_gen(QX,QY,TX,TY,WKX,WKY) :-
    DX = TX - QX, DY = TY - QY, DX = -DY, DX != 0,
    WDX = WKX - QX, WDY = WKY - QY, WDX = -WDY, WDX != 0,
    WDX < 0, DX < 0, WDX > DX.

% White king attacks square
wking_attacks_sq(WKX,WKY,TX,TY) :-
    coord(WKX), coord(WKY), coord(TX), coord(TY),
    king_delta(DX,DY),
    TX = WKX+DX, TY = WKY+DY.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% After white queen moves to (QTX,QTY):
% Can black king escape to (TX,TY)?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Black king candidate moves after queen move to QTX,QTY
bk_escape_candidate(QTX,QTY,TX,TY) :-
    after_queen_move(QTX,QTY,QTX,QTY,WKX,WKY,BKX,BKY),
    king_delta(DX,DY),
    TX = BKX+DX, TY = BKY+DY,
    square(TX,TY),
    % not occupied by black piece (no black pieces other than king)
    % can't move to white king square
    not (TX = WKX, TY = WKY).

% Black king escape is blocked if the target square is attacked by white
bk_escape_sq_attacked(QTX,QTY,TX,TY) :-
    bk_escape_candidate(QTX,QTY,TX,TY),
    after_queen_move(QTX,QTY,QTX,QTY,WKX,WKY,_,_),
    % If black king moves to TX,TY (not capturing queen), queen still at QTX,QTY
    not (TX = QTX, TY = QTY),
    queen_attacks_sq(QTX,QTY,WKX,WKY,TX,TY).

bk_escape_sq_attacked(QTX,QTY,TX,TY) :-
    bk_escape_candidate(QTX,QTY,TX,TY),
    after_queen_move(QTX,QTY,QTX,QTY,WKX,WKY,_,_),
    % If black king captures queen at QTX,QTY, queen gone but white king still attacks?
    TX = QTX, TY = QTY,
    wking_attacks_sq(WKX,WKY,TX,TY).

bk_escape_sq_attacked(QTX,QTY,TX,TY) :-
    bk_escape_candidate(QTX,QTY,TX,TY),
    after_queen_move(QTX,QTY,QTX,QTY,WKX,WKY,_,_),
    not (TX = QTX, TY = QTY),
    wking_attacks_sq(WKX,WKY,TX,TY).

% Black king has a legal escape after queen move to QTX,QTY
bk_has_escape_after_qmove(QTX,QTY) :-
    bk_escape_candidate(QTX,QTY,TX,TY),
    not bk_escape_sq_attacked(QTX,QTY,TX,TY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mate in one via queen move
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% White queen moves from (QFX,QFY) to (QTX,QTY), giving check with no escape

% Original queen position
piece(white,queen,4,7).

mate_move_queen(4,7,QTX,QTY) :-
    move_white(queen,4,7,QTX,QTY),
    bk_in_check_after_qmove(QTX,QTY),
    not bk_has_escape_after_qmove(QTX,QTY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Also handle white king moves (less likely to give mate here but for completeness)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% After white king moves to (KTX,KTY): queen at (QX,QY), white king at (KTX,KTY), black king at (BKX,BKY)
% Check if black king is in check
bk_in_check_after_kmove(KTX,KTY) :-
    after_king_move(KTX,KTY,QX,QY,KTX,KTY,BKX,BKY),
    % queen attacks black king with white king now at KTX,KTY
    queen_attacks_sq(QX,QY,KTX,KTY,BKX,BKY).

bk_in_check_after_kmove(KTX,KTY) :-
    after_king_move(KTX,KTY,_,_,KTX,KTY,BKX,BKY),
    wking_attacks_sq(KTX,KTY,BKX,BKY).

% Black king escape after white king move to KTX,KTY
bk_escape_candidate_km(KTX,KTY,TX,TY) :-
    after_king_move(KTX,KTY,QX,QY,KTX,KTY,BKX,BKY),
    king_delta(DX,DY),
    TX = BKX+DX, TY = BKY+DY,
    square(TX,TY),
    not (TX = KTX, TY = KTY).

bk_escape_sq_attacked_km(KTX,KTY,TX,TY) :-
    bk_escape_candidate_km(KTX,KTY,TX,TY),
    after_king_move(KTX,KTY,QX,QY,KTX,KTY,_,_),
    queen_attacks_sq(QX,QY,KTX,KTY,TX,TY).

bk_escape_sq_attacked_km(KTX,KTY,TX,TY) :-
    bk_escape_candidate_km(KTX,KTY,TX,TY),
    after_king_move(KTX,KTY,_,_,KTX,KTY,_,_),
    wking_attacks_sq(KTX,KTY,TX,TY).

bk_has_escape_after_kmove(KTX,KTY) :-
    bk_escape_candidate_km(KTX,KTY,TX,TY),
    not bk_escape_sq_attacked_km(KTX,KTY,TX,TY).

mate_move_king(7,6,KTX,KTY) :-
    move_white(king,7,6,KTX,KTY),
    bk_in_check_after_kmove(KTX,KTY),
    not bk_has_escape_after_kmove(KTX,KTY).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mate_move(white,queen,FX,FY,TX,TY) :- mate_move_queen(FX,FY,TX,TY).
mate_move(white,king,FX,FY,TX,TY) :- mate_move_king(FX,FY,TX,TY).

#show mate_move/6.